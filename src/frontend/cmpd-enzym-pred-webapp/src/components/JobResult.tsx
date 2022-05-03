// React Router
import {
  Outlet,
  Link as RouterLink,
  useLocation,
  matchPath,
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";

// UI components
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import { Fab, Grid, Stack, TextField } from "@mui/material";

// Data grid
import { DataGrid, GridColDef, GridValueGetterParams } from "@mui/x-data-grid";

// Icons
import HomeIcon from "@mui/icons-material/Home";
import AddIcon from "@mui/icons-material/Add";

// ID generation
import { v4 as uuidv4 } from "uuid";
import FileUpload from "./FileUpload";
import { useEffect, useState } from "react";

// http request
import axios from "axios";

// base URL for the backend. here we assume that the backend is running on the same machine
const baseUrl = `http://${window.location.hostname}:8004`;

interface JobResultProps {
  jobId: string | undefined;
}

export default function JobResult(props: JobResultProps) {
  const [jobName, setJobName] = useState("");
  const [jobSubmitter, setJobSubmitter] = useState("");
  const [jobEmail, setJobEmail] = useState("");

  function loadJob() {
    if (props.jobId) {
      axios
        .get(`${baseUrl}/get_job_status?job_id=${props.jobId}`)
        .then((res) => {
          setJobName(res.data.job_name);
          setJobSubmitter(res.data.submitter);
          setJobEmail(res.data.submitter_email);
        });
    }
  }

  useEffect(() => {
    loadJob();
  }, []);

  return (
    <Box component="form" noValidate>
      <Typography variant="h5">Result</Typography>
      <Typography variant="body1">Result and status of your job.</Typography>

      <Grid container spacing={2} marginTop={4}>
        <Grid item xs={12}>
          <TextField
            disabled
            label="Job ID"
            helperText="This ID is generated automatically"
            defaultValue={props.jobId}
            fullWidth
          ></TextField>
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField
            value={jobName}
            label="Job name"
            variant="outlined"
            InputProps={{
              readOnly: true,
            }}
            fullWidth
          ></TextField>
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField
            label="Submitter"
            value={jobSubmitter}
            variant="outlined"
            InputProps={{
              readOnly: true,
            }}
            fullWidth
          ></TextField>
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField
            label="Email"
            value={jobEmail}
            variant="outlined"
            InputProps={{
              readOnly: true,
            }}
            fullWidth
          ></TextField>
        </Grid>
      </Grid>
    </Box>
  );
}
