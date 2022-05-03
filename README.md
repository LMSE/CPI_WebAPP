# EAP-Web (Enzyme Activity Prediction web app)

## Backend
To run the backend, build and run the Docker containers. Navigate to the root directory of this repo and run the following commands.

```bash
docker-compose build
docker-compose up
```

## Frontend

Frontend is built using React. To run the frontend in a development environment, navigate to the backend folder and run
```bash
npm start
```

## Resource Files
Some resource files used by this web application is quite large (especially the model files). The large files are stored on Git LFS. To see more on how to use and pull files from Git LFS, check https://git-lfs.github.com/
