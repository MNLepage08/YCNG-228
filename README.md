# YCNG228
## Predictive &amp; Classification Modelling

1. Overview of the different phases of a data science project. Technologies used in this project. Definition of the project. Agile methodology. General organization.

2. Definition of the objectives. How to scope a project. Presentation of the data used for this project.

3. Definition of roles and responsibilities. What are the different skills required to develop the solution?

4. Sprint 0: Set the stage to develop the solution.

5. Sprint 1: get a baseline, design of experiments, hypothesis testing. 

6. Sprint 2: Productize the baseline.

7. Iteration 1: Improve the solution. Methods to increase accuracy/ precision, or other metrics. How to optimize your time?

8. End of the exploration and long-term considerations. 

9. What can go wrong? Common pitfall and how to avoid it. 

10. Manage Bias in models and explainable AI.

11. Final exam.

12. Retrospective.

## Conda
  - [Installation of conda: ](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)The fastest way to obtain conda is to install Miniconda, a mini version of Anaconda that includes only conda and its dependencies.

## Docker
  - [Installation of docker: ](https://docs.docker.com/get-docker/)Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production.
  - Build a docker image on local machine:
```diff
docker build . -f Dockerfile -t my_image_v2      
```
  - Run the docker image:
```diff
docker run -p 8081:8081 -v $GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json my_image_v2     
```
  - Test if the app is working:
```diff
http://0.0.0.0/[name_of_your_end_point] 
```
  - (Optional) If for some reasons, you want to see what is going on inside the docker, you can start it in an interacting mode:
```diff
docker run -it -p 8081:8081 -v $GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json my_image_v2 /bin/bash    
```

## GitHub
  - [Create a repository: ](https://docs.github.com/en/get-started/quickstart/create-a-repo) You can store a variety of projects in GitHub repositories, including open source projects. With open source projects, you can share code to make better, more reliable software. You can use repositories to collaborate with others and track your work.
  
## Google Cloud Platform
  - [Create your project: ](https://cloud.google.com/resource-manager/docs/creating-managing-projects)Google Cloud projects form the basis for creating, enabling, and using all Google Cloud services including managing APIs, enabling billing, adding and removing collaborators, and managing permissions for Google Cloud resources.
  - [Get the credentials (json): ](https://cloud.google.com/docs/authentication/client-libraries)To use Application Default Credentials to authenticate your application, you must first set up Application Default Credentials for the environment where your application is running. When you use the client library to create a client, the client library automatically checks for and uses the credentials you have provided to ADC to authenticate to the APIs your code uses. Your application does not need to explicitly authenticate or manage tokens; these requirements are managed automatically by the authentication libraries.
```diff
export GOOGLE_APPLICATION_CREDENTIALS='/path of the credentials.json'         
```
  - [Set up CI/CD: ](https://cloud.google.com/build/docs/automating-builds/github/build-repos-from-github) GitHub triggers enable you to automatically build on Git pushes and pull requests and view your build results on GitHub and Google Cloud console. Additionally, GitHub triggers support all the features supported by the existing GitHub triggers and use the Cloud Build GitHub app to configure and authenticate to GitHub.

## On the machine
  - Clone the GitHub
  - [Create the environement ](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)with scripts/environment.yml
  - Activate the environment: conda activate stock
  - Run the app: python app.py
  - Test the endpoint with a separate shell: http://localhost:8081/

## Code
  - `application.conf:` Contains the parametrisation of the app. Used in the code to load constant like the version number of the app.
  - `Dockerfile:` Contains the definition of the steps to create the docker image. The image will be created by google build (CI/CD) and saved into the google storage. Use this file to test the docker image on your local machine. See the section "Build and test the docker image".
  - `get_data.py:` Use this script to download the stock history for S&P500
  - `app.py:` Contains the main for the Flask server. It is also the entrypoint of the app. 
  - `build_and_deploy_docker_image.sh:` Contains some basic instruction on how to build a docker image and running it.
  - `src/algo:` This directory contains the code to fit and predict stock with a model.
  - `src/business_logic:` This code contains the logic to process the query, deal with model storage etc...
  - `src/IO:` This code deal with fetching the data, accessing the google storage etc...

## Bibliography
* [Desing Thinking](https://readings.design/PDF/Tim%20Brown,%20Design%20Thinking.pdf)
