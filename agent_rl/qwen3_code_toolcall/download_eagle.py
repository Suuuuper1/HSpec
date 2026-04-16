# Copyright 2025 Chinese Information Processing Laboratory, ISCAS.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-1.7B"
EAGLE_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"


def download(model_id):
    logger.info("Downloading %s ...", model_id)
    folder_name = model_id.replace("/", "--")
    path = snapshot_download(
        repo_id=model_id,
        resume_download=True,
        local_dir_use_symlinks=False,
        local_dir=f"./models/{folder_name}",
    )
    logger.info("Saved to: %s", path)
    return path

if __name__ == "__main__":
    eagle_path = download(EAGLE_MODEL)
    base_path = download(BASE_MODEL)

    logger.info("All models downloaded successfully:")
    logger.info("Base model: %s", base_path)
    logger.info("Eagle model: %s", eagle_path)