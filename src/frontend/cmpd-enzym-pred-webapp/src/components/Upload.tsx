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
import { Fab, Grid, Link, Stack, TextField } from "@mui/material";

// Data grid
import { DataGrid, GridColDef, GridValueGetterParams } from "@mui/x-data-grid";

// Icons
import HomeIcon from "@mui/icons-material/Home";
import AddIcon from "@mui/icons-material/Add";

// ID generation
import { v4 as uuidv4 } from "uuid";
import FileUpload from "./FileUpload";
import { useState } from "react";

interface UploadProps {
  seqFile?: File;
  subFile?: File;
  jobId: string;
  jobName: string;
  jobSubmitter: string;
  jobSubmitterEmail: string;
  onSeqUpload?: (file: File) => void;
  onSubUpload?: (file: File) => void;
  onJobIdChange?: (e: string) => void;
  onJobNameChange?: (e: string) => void;
  onJobSubmitterChange?: (e: string) => void;
  onJobEmailChange?: (e: string) => void;
}

export default function Upload(props: UploadProps) {
  const [seqFile, setSeqFile] = useState<File | undefined>(undefined);
  const [subFile, setSubFile] = useState<File | undefined>(undefined);

  return (
    <Box component="form" noValidate>
      <Typography variant="h5">Upload and Preprocess</Typography>
      <Typography variant="body1">
        Upload your sequence and substrate data and preprocess it. All data
        should be prepared using the pipeline available at
        <br />
        <Link href="https://github.com/LMSE/CmpdEnzymPred">
          github.com/LMSE/CmpdEnzymPred
        </Link>
      </Typography>

      <Grid container spacing={2} marginTop={4}>
        <Grid item xs={12}>
          <TextField
            disabled
            label="Job ID"
            helperText="This ID is generated automatically"
            defaultValue={uuidv4()}
            fullWidth
            onChange={(e) => {
              if (props.onJobIdChange) {
                props.onJobIdChange(e.target.value);
              }
            }}
          ></TextField>
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField
            label="Job name"
            fullWidth
            onChange={(e) => {
              if (props.onJobNameChange) {
                props.onJobNameChange(e.target.value);
              }
            }}
          />
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField
            label="Submitter"
            fullWidth
            onChange={(e) => {
              if (props.onJobSubmitterChange) {
                props.onJobSubmitterChange(e.target.value);
              }
            }}
          />
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField
            label="Email"
            fullWidth
            onChange={(e) => {
              if (props.onJobEmailChange) {
                props.onJobEmailChange(e.target.value);
              }
            }}
          />
        </Grid>
        <Grid item md={6} sm={12} marginTop={4}>
          <FileUpload
            inputId="seq-upload"
            text="Upload sequence embedding"
            onUpload={(e) => {
              if (props.onSeqUpload) {
                props.onSeqUpload(e);
                setSeqFile(e);
              }
            }}
            file={seqFile}
          ></FileUpload>
        </Grid>
        <Grid item md={6} sm={12} marginTop={4}>
          <FileUpload
            inputId="substrate-upload"
            text="Upload substrate encoding"
            onUpload={(e) => {
              if (props.onSubUpload) {
                props.onSubUpload(e);
                setSubFile(e);
              }
            }}
            file={subFile}
          ></FileUpload>
        </Grid>
      </Grid>
    </Box>
  );
}
