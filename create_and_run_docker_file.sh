#for local testing, you can build and run your docker image locally
docker build . -f Dockerfile -t my_image_v2

#to run the docker image:
docker run -p 8081:8081 -v $GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json my_image_v2
#to log into the docker image:
docker run -it -p 8081:8081 -v $GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json my_image_v2 /bin/bash
