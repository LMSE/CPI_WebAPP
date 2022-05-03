import * as React from "react";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Typography from "@mui/material/Typography";
import { Box, CardActionArea } from "@mui/material";

import CancelIcon from "@mui/icons-material/Cancel";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";

interface FileUploadProps {
  inputId: string;
  file?: File;
  onUpload?: (file: File) => void;
  text?: string;
}

export default function FileUpload(props: FileUploadProps) {
  const [file, setFile] = React.useState<File | undefined>(undefined);
  return (
    <div>
      <input
        type="file"
        id={props.inputId}
        onChange={(e) => {
          if (props.onUpload) {
            if (e.target && e.target.files) {
              props.onUpload(e.target.files[0]);
              setFile(e.target.files[0]);
            }
          }
        }}
        hidden
      />
      <label htmlFor={props.inputId}>
        <Card
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (props.onUpload) {
              props.onUpload(file);
            }
          }}
        >
          <CardContent>
            <Box
              height={140}
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {file ? <CheckCircleIcon style={{ marginRight: 8 }} /> : ""}
              <Typography variant="body2" color="text.secondary">
                {props.text ? props.text : "Upload file"}
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </label>
    </div>
  );
}
