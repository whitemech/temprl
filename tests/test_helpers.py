#
# Copyright 2020-2022 Marco Favorito
#
# ------------------------------
#
# This file is part of temprl.
#
# temprl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# temprl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with temprl.  If not, see <https://www.gnu.org/licenses/>.
#

"""Tests for `temprl.helpers` module."""
import pytest

from temprl.helpers import enforce


def test_enforce_negative() -> None:
    """Test 'enforce' raises exception."""
    error_message = "some error message here"
    with pytest.raises(ValueError, match=error_message):
        enforce(1 == 2, error_message, exception_cls=ValueError)
