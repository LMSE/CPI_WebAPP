import * as React from "react";
import Box from "@mui/material/Box";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { Fab, Fade, Grow, Stack, Zoom } from "@mui/material";
import { Router, useLocation, useNavigate } from "react-router-dom";

// internal components
import Upload from "../components/Upload";
import EditParameters from "../components/EditParameters";

// uuid generation
import { v4 as uuidv4 } from "uuid";
// http request
import axios from "axios";

const steps = ["Upload files", "Model parameters", "Submit job"];

// base URL for the backend. here we assume that the backend is running on the same machine
const baseUrl = `http://${window.location.hostname}:8004`;

export default function NewJobView() {
  const [activeStep, setActiveStep] = React.useState(0);
  const [skipped, setSkipped] = React.useState(new Set<number>());
  let location = useLocation();
  let navigate = useNavigate();

  const [jobId, setJobId] = React.useState(uuidv4());
  const [jobName, setJobName] = React.useState("");
  const [jobSubmitter, setJobSubmitter] = React.useState("");
  const [jobEmail, setJobEmail] = React.useState("");
  const [jobSeqFile, setJobSeqFile] = React.useState<File | undefined>(
    undefined
  );
  const [jobSubFile, setJobSubFile] = React.useState<File | undefined>(
    undefined
  );

  const isStepOptional = (step: number) => {
    return step === 1;
  };

  const isStepSkipped = (step: number) => {
    return skipped.has(step);
  };

  const handleNext = () => {
    let newSkipped = skipped;
    if (isStepSkipped(activeStep)) {
      newSkipped = new Set(newSkipped.values());
      newSkipped.delete(activeStep);
    }

    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setSkipped(newSkipped);

    // last step, submit job
    if (activeStep === steps.length - 1) {
      // create the job record in the database
      axios.get(`${baseUrl}/create_job`, {
        params: {
          job_id: jobId,
          job_name: jobName,
          submitter: jobSubmitter,
          submitter_email: jobEmail,
        },
      });
      // upload files
      if (jobSeqFile && jobSubFile) {
        const seqFormData = new FormData();
        seqFormData.append("upload", jobSeqFile);
        axios.post(
          `${baseUrl}/file/upload?job_id=${jobId}&job_type=seq`,
          seqFormData
        );

        const subFormData = new FormData();
        subFormData.append("upload", jobSubFile);
        axios.post(
          `${baseUrl}/file/upload?job_id=${jobId}&job_type=sub`,
          subFormData
        );
      }
      // run the job
      axios.get(`${baseUrl}/run_prediction?job_id=${jobId}`);
      // return to home
      navigate("/");
    }
  };

  const handleBack = () => {
    if (activeStep === 0) {
      // redirect to home
      navigate("/");
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep - 1);
    }
  };

  const handleSkip = () => {
    if (!isStepOptional(activeStep)) {
      // You probably want to guard against something like this,
      // it should never occur unless someone's actively trying to break something.
      throw new Error("You can't skip a step that isn't optional.");
    }

    setActiveStep((prevActiveStep) => prevActiveStep + 1);
    setSkipped((prevSkipped) => {
      const newSkipped = new Set(prevSkipped.values());
      newSkipped.add(activeStep);
      return newSkipped;
    });
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  const getStepContent = (step: number) => {
    switch (activeStep) {
      case 0:
        return (
          <Upload
            jobId={jobId}
            jobName={jobName}
            jobSubmitter={jobSubmitter}
            jobSubmitterEmail={jobEmail}
            seqFile={jobSeqFile}
            subFile={jobSubFile}
            onJobNameChange={(e) => setJobName(e)}
            onJobSubmitterChange={(e) => setJobSubmitter(e)}
            onJobEmailChange={(e) => setJobEmail(e)}
            onSeqUpload={onSeqUpload}
            onSubUpload={onSubUpload}
          />
        );
      case 1:
        // TODO: add edit parameters page
        return <EditParameters />;
      default:
        // TODO: change this to an overview page
        return (
          <Upload
            jobId={jobId}
            jobName={jobName}
            jobSubmitter={jobSubmitter}
            jobSubmitterEmail={jobEmail}
            seqFile={jobSeqFile}
            subFile={jobSubFile}
          />
        );
    }
  };

  function onSeqUpload(file: File | null) {
    if (file) setJobSeqFile(file);
    console.log(file?.name);
  }

  function onSubUpload(file: File | null) {
    if (file) setJobSubFile(file);
    console.log(file?.name);
  }

  return (
    <Grow in={true} unmountOnExit>
      <Stack sx={{ width: "100%", height: "100vh", alignItems: "stretch" }}>
        <Stepper activeStep={activeStep} sx={{ margin: 4 }}>
          {steps.map((label, index) => {
            const stepProps: { completed?: boolean } = {};
            const labelProps: {
              optional?: React.ReactNode;
            } = {};
            if (isStepOptional(index)) {
              labelProps.optional = (
                <Typography variant="caption">Optional</Typography>
              );
            }
            if (isStepSkipped(index)) {
              stepProps.completed = false;
            }
            return (
              <Step key={label} {...stepProps}>
                <StepLabel {...labelProps}>{label}</StepLabel>
              </Step>
            );
          })}
        </Stepper>
        <Box sx={{ height: "100%", margin: 4 }}>
          {activeStep === steps.length ? (
            <Stack style={{ height: "100%" }}>
              <Box style={{ height: "90%" }}>
                <Typography sx={{ mt: 2, mb: 1 }}>
                  All steps completed - you&apos;re finished
                </Typography>
              </Box>
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "row",
                  pt: 2,
                  height: "10%",
                }}
              >
                <Box sx={{ flex: "1 1 auto" }} />
                <Button onClick={handleReset} sx={{ height: 37 }}>
                  Reset
                </Button>
              </Box>
            </Stack>
          ) : (
            <Stack style={{ height: "100%" }}>
              <Box style={{ height: "90%", margin: 24 }}>
                {getStepContent(activeStep)}
              </Box>
              <Box sx={{ display: "flex", flexDirection: "row", pt: 2 }}>
                <Button color="inherit" onClick={handleBack} sx={{ mr: 1 }}>
                  Back
                </Button>
                <Box sx={{ flex: "1 1 auto" }} />
                {isStepOptional(activeStep) && (
                  <Button color="inherit" onClick={handleSkip} sx={{ mr: 1 }}>
                    Skip
                  </Button>
                )}
                <Button onClick={handleNext}>
                  {activeStep === steps.length - 1 ? "Finish" : "Next"}
                </Button>
              </Box>
            </Stack>
          )}
        </Box>
      </Stack>
    </Grow>
  );
}
