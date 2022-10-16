# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from continuumio/miniconda3
# Allow statements and log messages to immediately appear in the Knative logs
env PYTHONUNBUFFERED True
# Copy local code to the container image.
env APP_HOME /app
workdir $APP_HOME
copy . ./


run conda env create -f scripts/environment.yml
shell ["conda", "run", "-n", "stock", "/bin/bash", "-c"]
run pip install flask gunicorn

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
env PORT 8081
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
cmd exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
