# Copyright (C) 2024-2025 Guyutongxue
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import threading
from ._threading import ensure_thread_initialized, initialize_library, thread_cleanup as _thread_cleanup, thread_initialize as _thread_initialize

initialize_library()
if threading.current_thread() is threading.main_thread():
    ensure_thread_initialized()

def thread_initialize():
    """
    If running gitcg in a non-main thread, call this function to initialize the thread.
    """
    _thread_initialize()

def thread_cleanup():
    """
    If running gitcg in a non-main thread, call this function to clean up the thread.
    """
    _thread_cleanup()
