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
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import { Fab, Grid, Stack, TextField } from "@mui/material";

// Data grid
import { DataGrid, GridColDef, GridValueGetterParams } from "@mui/x-data-grid";

// Icons
import HomeIcon from "@mui/icons-material/Home";
import AddIcon from "@mui/icons-material/Add";
import BatchPredictionIcon from "@mui/icons-material/BatchPrediction";

// ID generation
import { v4 as uuidv4 } from "uuid";
import FileUpload from "./FileUpload";

export default function EditParameters() {
  return (
    <Box component="form" noValidate>
      <Typography variant="h5">Parameters and Additional Info</Typography>
      <Typography variant="body1">
        Below are the parameters inferred from your data. You can edit them if
        any of them is incorrect.
      </Typography>

      <Grid container spacing={2} marginTop={4}>
        <Grid item xs={12}>
          <TextField
            disabled
            label="Job ID"
            helperText="This ID is generated automatically"
            defaultValue={uuidv4()}
            fullWidth
          ></TextField>
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField label="Sequence number" fullWidth />
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField label="Substrate number" fullWidth />
        </Grid>
        <Grid item md={4} sm={6}>
          <TextField label="Max sequence length" fullWidth />
        </Grid>
        <Grid item>
          <Stack spacing={1}>
            <Typography color="GrayText">Select a model</Typography>
            <ToggleButtonGroup exclusive aria-label="Select Model">
              <ToggleButton value="center" aria-label="centered">
                <BatchPredictionIcon style={{ marginRight: 4 }} /> Model 1
              </ToggleButton>
              <ToggleButton value="center" aria-label="centered">
                <BatchPredictionIcon style={{ marginRight: 4 }} /> Model 2
              </ToggleButton>
            </ToggleButtonGroup>
          </Stack>
        </Grid>
      </Grid>
    </Box>
  );
}
