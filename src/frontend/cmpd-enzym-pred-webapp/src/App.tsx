// React Router
import {
  Outlet,
  Link as RouterLink,
  useLocation,
  matchPath,
  BrowserRouter,
  Routes,
  Route,
  useNavigate,
} from "react-router-dom";

// UI components
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import CancelIcon from "@mui/icons-material/Cancel";
import HourglassBottomIcon from "@mui/icons-material/HourglassBottom";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import { Fab, Stack, Grow } from "@mui/material";

// Data grid
import {
  DataGrid,
  GridColDef,
  GridValueGetterParams,
  GridCell,
  GridCheckCircleIcon,
} from "@mui/x-data-grid";

// Icons
import HomeIcon from "@mui/icons-material/Home";
import AddIcon from "@mui/icons-material/Add";

// ID generation
import { v4 as uuidv4 } from "uuid";
import { useEffect, useState } from "react";
import axios from "axios";

// base URL for the backend. here we assume that the backend is running on the same machine
const baseUrl = `http://${window.location.hostname}:8004`;

const statuses: string[] = ["Running", "Completed", "Failed"];

const columns: GridColDef<Job>[] = [
  { field: "id", flex: 0.1, headerName: "ID" },
  {
    field: "submittedBy",
    headerName: "Submitted by",
    flex: 1,
  },
  {
    field: "jobName",
    headerName: "Job name",
    flex: 1,
  },
  {
    field: "jobId",
    headerName: "Job ID",
    type: "string",
    width: 200,
    flex: 1,
  },
  {
    field: "jobStatus",
    headerName: "Job status",
    type: "string",
    sortable: true,
    flex: 0.5,
    renderCell: (params: GridValueGetterParams) => {
      let icon = (
        <HourglassBottomIcon
          color="primary"
          style={{ marginRight: 4 }}
          fontSize="small"
        />
      );
      if (params.value === "Failed") {
        icon = (
          <CancelIcon
            color="error"
            style={{ marginRight: 4 }}
            fontSize="small"
          />
        );
      } else if (params.value === "Completed") {
        icon = (
          <GridCheckCircleIcon
            color="success"
            style={{ marginRight: 4 }}
            fontSize="small"
          />
        );
      }
      return (
        <Stack flexDirection="row" alignItems="center" spacing={2}>
          {icon}
          {params.row.jobStatus}
        </Stack>
      );
    },
  },
  {
    field: "time",
    headerName: "Time submitted",
    flex: 1,
    sortable: true,
    // valueGetter: (params: GridValueGetterParams) =>
    //   `${params.row.firstName || ""} ${params.row.lastName || ""}`,
  },
];

interface Job {
  id: number;
  jobId: string;
  jobName: string;
  jobStatus: string;
  submittedBy: string;
  time: string;
}

const defaultRows: Job[] = [];

export default function App() {
  const fabStyle = {
    position: "absolute",
    bottom: 48,
    right: 48,
  };

  function loadJobs(): Promise<Job[]> {
    return new Promise((resolve) => {
      setTimeout(() => {
        axios.get(`${baseUrl}/get_all_jobs`).then((res) => {
          let parsed_data: Job[] = [];
          for (let i = 0; i < res.data.length; i++) {
            parsed_data.push({
              id: i + 1,
              jobId: res.data[i].job_id,
              jobName: res.data[i].job_name,
              jobStatus: statuses[res.data[i].status],
              submittedBy: res.data[i].submitter,
              time: res.data[i].time_created,
            });
            console.log(res.data[i]);
          }
          resolve(parsed_data);
        });
      }, 1000);
    });
  }

  const [jobs, setJobs] = useState<Job[]>(defaultRows);

  const navigate = useNavigate();

  useEffect(() => {
    const j = loadJobs()
      .catch(console.error)
      .then((j) => {
        if (j) {
          setJobs(j);
        }
      });
  }, []);

  return (
    <Grow in={true} unmountOnExit>
      <Stack height="100vh">
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              EAP-ML Online
            </Typography>
          </Toolbar>
        </AppBar>
        <Fab sx={fabStyle} color="primary" component={RouterLink} to="/newjob">
          <AddIcon />
        </Fab>
        <div style={{ height: "100%", width: "100%" }}>
          <DataGrid
            rows={jobs}
            columns={columns}
            pageSize={25}
            rowsPerPageOptions={[5]}
            checkboxSelection
            disableSelectionOnClick
            onRowClick={(params, e, details) => {
              console.log(params.row.jobId);
              navigate(`/job/${params.row.jobId}`);
            }}
          />
        </div>
      </Stack>
    </Grow>
  );
}
