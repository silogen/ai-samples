# Copyright 2026 Advanced Micro Devices, Inc.  All rights reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
#       http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nerfstudio.cameras.cameras import Cameras as NsCameras


class Cameras(NsCameras):
    """
    We override nerfstudio's Cameras data structure to provide a cleaner
    handling of timestamps for the purposes of indexing annotations.
    It avoids (a) copying values back and forther between GPU and main memory
    unnecessarily and (b) messy conversions between different timestamp formats
    during processing.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_times_ms()

    def update_times_ms(self):
        self._times_ms = [int(t.item()*1000) for t in self.times]

    @property
    def times_ms(self):
        """
        Timestamps as integers, stored in main memory (not GPU).

        Using this avoids copying from the GPU every time you want the timestamp
        and gives a timestamp with fixed precision (integer milliseconds) that can
        be used for indexing.

        However, to avoid those GPU copies, the stored values are only computed
        from `self.times` at init and when `camera.update_times_ms()` is called.
        Most of the time, the times are invariant, so this is fine, but be sure
        to call the update method is times get changed anywhere.

        """
        return self._times_ms
    
    @property
    def time_ms(self):
        """
        Same as `times_ms` but always returns the first value.

        We use `Cameras` to store only a single camera, so this is just more readable
        than always calling `camera.times_ms[0]`.
        
        """
        return self._times_ms[0]
