# EAP-Web (Enzyme Activity Prediction web app)

## Pulling the Source Code

The repository includes multiple large files that cannot be stored directly on GitHub. Hence, we used Git LFS. To pull those files properly, make sure to have Git LFS installed before pulling the repository.

```bash
git lfs install
```

If the repository is cloned before installing Git LFS, the files will show up in your directory as a placeholder. In this case, run

```bash
git checkout .
```

after installing Git LFS.

## Backend
To run the backend, build and run the Docker containers. Navigate to the root directory of this repo and run the following commands. Install the latest Docker version at https://www.docker.com/get-started/ and follow the instructions. Note that we need to have Docker running in the background before running these commands.

```bash
docker-compose build
docker-compose up
```
Once we finish running these two commands, leave the command window open, and open a new window for the remaining commands.

## Frontend

Frontend is built using React and required Node.JS. To install Node, see https://nodejs.org/en/ and download the LTS version.

To run the frontend in a development environment, navigate to the frontend folder and run
```bash
npm install
npm start
```

## Using the App

After running `npm start` and starting the backend in Docker containers. You should see a web page opening up in your browser window. This is the main dashboard of the application. Click the add (plus sign) button at the lower-right corner to add a new job. Input the job information and upload the required files. The files should be `.p` files created using the pipeline at https://github.com/LMSE/CmpdEnzymPred. Skip the second and last steps as those are just placeholders for now. Submit the job. After submission, you will be brought back to the dashboard. Your newly submitted job should appear in the list as "Running". Wait for a few seconds (or a minute depending on the device you are using to test this) and refresh the dashboard. When you see that the job is marked as "Completed", click on it to see the results.

## Resource Files
Some resource files used by this web application is quite large (especially the model files). The large files are stored on Git LFS. To see more on how to use and pull files from Git LFS, check https://git-lfs.github.com/
