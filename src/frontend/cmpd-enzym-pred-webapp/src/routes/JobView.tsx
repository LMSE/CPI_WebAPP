import * as React from "react";

import Box from "@mui/material/Box";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";

// icons
import DeleteForeverIcon from "@mui/icons-material/DeleteForever";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";

import { Fab, Fade, Grid, Grow, Stack, Zoom } from "@mui/material";
import {
  Router,
  useLocation,
  useNavigate,
  useParams,
  Link as RouterLink,
} from "react-router-dom";

// internal components
import Upload from "../components/Upload";
import JobResult from "../components/JobResult";
// Data grid
import {
  DataGrid,
  GridColDef,
  GridValueGetterParams,
  GridCell,
  GridCheckCircleIcon,
} from "@mui/x-data-grid";

import axios from "axios";
import { ReactNode, useEffect, useState } from "react";

// base URL for the backend. here we assume that the backend is running on the same machine
const baseUrl = `http://${window.location.hostname}:8004`;

const steps = [
  "Upload and preprocess files",
  "Training parameters",
  "Submit job",
];

interface PredictionResult {
  seq: string;
  sub: string;
  val: number;
}

export default function JobView() {
  let location = useLocation();
  let navigate = useNavigate();
  let params = useParams();

  const [predictionResults, setPredictionResults] = useState<
    PredictionResult[]
  >([]);

  const [jobStatus, setJobStatus] = useState(0);

  const columns: GridColDef<PredictionResult>[] = [
    {
      field: "seq",
      headerName: "Sequence",
      flex: 1,
    },
    {
      field: "sub",
      headerName: "Substrate",
      flex: 1,
    },
    {
      field: "val",
      headerName: "Prediction",
      flex: 1,
    },
  ];

  function loadParseResults() {
    axios.get(`${baseUrl}/get_result?job_id=${params.jobId}`).then((res) => {
      const parsedData = JSON.parse(res.data);
      setPredictionResults(parsedData);
    });
  }

  const fabStyle = {
    position: "absolute",
    bottom: 48,
    right: 48,
  };

  const fabStyleLeft = {
    position: "absolute",
    bottom: 48,
    left: 48,
  };

  useEffect(() => {
    axios
      .get(`${baseUrl}/get_job_status?job_id=${params.jobId}`)
      .then((res) => {
        setJobStatus(res.data.status);
      });
    loadParseResults();
  }, []);

  return (
    <Grow in={true} unmountOnExit>
      <Stack
        sx={{
          width: "100%",
          height: "100vh",
          alignItems: "stretch",
        }}
      >
        <Stack margin={8} height="100%">
          <JobResult jobId={params.jobId} />
          {jobStatus == 1 ? (
            <Grid container spacing={2} marginTop={4} height="100%">
              <Grid item xs={6}>
                <DataGrid
                  columns={columns}
                  rows={predictionResults}
                  getRowId={(row) => row.seq + row.sub}
                ></DataGrid>
              </Grid>
              <Grid item xs={6}>
                <img
                  src={`${baseUrl}/get_heatmap?job_id=${params.jobId}`}
                  style={{ width: "100%" }}
                ></img>
              </Grid>
            </Grid>
          ) : (
            <Typography style={{ marginTop: 8 }}>
              Job is still running.
            </Typography>
          )}
        </Stack>
        <Fab sx={fabStyle} color="error">
          <DeleteForeverIcon />
        </Fab>
        <Fab sx={fabStyleLeft} color="primary" component={RouterLink} to="/">
          <ArrowBackIcon />
        </Fab>
      </Stack>
    </Grow>
  );
}
