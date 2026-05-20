# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import pytest

from llumnix.load_computation import (
    RemainingStepsLoad,
    KvBlocksRatioLoad,
    AdaptiveDecodeBatchLoad,
    MissWaitingTokensLoad,
)


class TestRemainingStepsLoadOrdering:
    """Regression tests for RemainingStepsLoad comparator (sort tiebreak fairness).

    Before this fix, RemainingStepsLoad.__lt__ used '>=' which returned True
    in BOTH directions when both operands were equal (e.g. both inf when all
    instances were idle). Combined with the missing __eq__, Python tuple-key
    sort '(load, instance_id)' could never fall through to instance_id, so
    sort order on idle ties was determined entirely by Timsort artifacts of
    the broken comparator instead of the intended UUID tiebreak. On a 5-GPU
    heterogeneous cluster this produced 76.7 / 17.3 / 5.2 / 0.78 / 0.0 percent
    dispatch skew and silently starved the last-registered instance.
    """

    def test_lt_is_strict_at_equal_values(self):
        a = RemainingStepsLoad(10.0)
        b = RemainingStepsLoad(10.0)
        # At equal values, AT MOST ONE direction of '<' should be True.
        assert not (a < b and b < a),             'broken strict weak ordering: both a<b and b<a return True'

    def test_lt_is_strict_at_inf(self):
        a = RemainingStepsLoad(math.inf)
        b = RemainingStepsLoad(math.inf)
        assert not (a < b and b < a),             'broken strict weak ordering at inf: both directions True'

    def test_eq_compares_by_value_not_identity(self):
        a = RemainingStepsLoad(10.0)
        b = RemainingStepsLoad(10.0)
        assert a == b, 'distinct objects with same load should compare equal'

    def test_eq_inf(self):
        assert RemainingStepsLoad(math.inf) == RemainingStepsLoad(math.inf)

    def test_hash_consistent_with_eq(self):
        a = RemainingStepsLoad(10.0)
        b = RemainingStepsLoad(10.0)
        assert hash(a) == hash(b)

    def test_descending_semantics_preserved(self):
        # The comparator intentionally inverts: higher remaining_steps sorts FIRST
        # (i.e., more idle instances are preferred). Verify this is still true.
        low = RemainingStepsLoad(1.0)
        high = RemainingStepsLoad(100.0)
        assert high < low, 'higher remaining_steps must sort before lower'
        assert not (low < high)

    def test_tuple_tiebreak_reaches_instance_id(self):
        # The actual call pattern from sort_instance_infos:
        # sorted(items, key=lambda i: (load, instance_id))
        # With the missing __eq__ pre-fix, two RemainingStepsLoad objects with
        # the same value were never equal, so the instance_id tiebreak was
        # unreachable. After the fix, ties on the load fall through to UUID.
        items = [
            (RemainingStepsLoad(math.inf), 'zzz_last'),
            (RemainingStepsLoad(math.inf), 'aaa_first'),
            (RemainingStepsLoad(math.inf), 'mmm_mid'),
        ]
        result = sorted(items)
        # Expected: when load is tied, second element (instance_id) tie-breaks
        # alphabetically ascending. So 'aaa_first' must come first.
        assert result[0][1] == 'aaa_first',             f'tuple tiebreak by instance_id failed: got order {[r[1] for r in result]}'
        assert result[-1][1] == 'zzz_last'

    def test_sort_does_not_silently_starve_first_input(self):
        # Regression for the production symptom: with 5 instances all idle and
        # broken comparator, Timsort emitted input-reverse on CPython 3.10 so
        # the FIRST-registered instance landed last and was never picked under
        # any dispatch policy that selects sorted[0]. After fix, with identical
        # load and instance_ids sorted alphabetically, the lexicographically
        # smallest UUID wins regardless of registration order.
        registration_order = ['e_last_registered', 'd', 'c', 'b', 'a_first_registered']
        items = [(RemainingStepsLoad(math.inf), uid) for uid in registration_order]
        result = sorted(items)
        first_pick = result[0][1]
        assert first_pick.startswith('a_'),             f'sorted[0] should be lexicographically smallest UUID, got {first_pick}'


class TestSiblingLoadClassesAlreadyStrict:
    """Sanity: sibling Load classes already use strict '<' and remain correct."""

    @pytest.mark.parametrize('cls,attr', [
        (KvBlocksRatioLoad, 'demand_factor'),
        (AdaptiveDecodeBatchLoad, 'decode_batch_size'),
        (MissWaitingTokensLoad, 'miss_waiting_tokens'),
    ])
    def test_sibling_lt_is_strict(self, cls, attr):
        a = cls(5.0)
        b = cls(5.0)
        assert not (a < b and b < a)
