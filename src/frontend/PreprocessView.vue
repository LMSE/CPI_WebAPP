<template>
  <div id="app">
    <div
      class="upload"
      @dragover.prevent
      @drop.prevent="(e) => onDrop(e, 'seq')"
    >
      Upload Seq
    </div>
    <div
      class="upload"
      @dragover.prevent
      @drop.prevent="(e) => onDrop(e, 'sub')"
    >
      Upload Sub
    </div>
    <button class="btn get-stat" @click="getStat">Get Stat</button>
    <div class="message">{{ message }}</div>
  </div>
</template>

<script lang="ts">
import { Options, Vue } from "vue-class-component";
import HelloWorld from "@/components/HelloWorld.vue"; // @ is an alias to /src

const host = "http://localhost:8004";

@Options({
  components: {
    HelloWorld,
  },
})
export default class HomeView extends Vue {
  message: string = "Welcome!";
  seqId: string | null = null;
  subId: string | null = null;

  async onDrop(e: DragEvent, type: string) {
    // Check file type
    const item = e.dataTransfer?.items[0];
    if (item == null || item.kind !== "file")
      return (this.message = `Error: The item dropped must be a file, not a ${item?.kind}`);

    const file = item?.getAsFile();
    if (file != null) {
      this.message = "File dropped, uploading...";
      console.log(
        `File Dropped: ${file?.name}\n` +
          `- LastModified: ${file?.lastModified}\n` +
          `- Size: ${file?.size}\n` +
          `- Type: ${file?.type}`
      );

      // Upload file for pre-processing
      let formData = new FormData();
      formData.append("upload", file);
      let res = await fetch(`${host}/file/upload?job_type=${type}`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok)
        return (this.message = `Error: Response not ok ${
          res.status
        } ${await res.text()}`);

      let fileId = (await res.json())["file_id"];
      this.message = `File upload success! File id is ${fileId}`;

      console.log(type);
      if (type === "seq") this.seqId = fileId;
      else this.subId = fileId;
    }
  }

  async getStat() {
    if (!this.seqId)
      return (this.message = `Error: You must upload sequence first`);

    this.message = "Sending request...";
    let res = await fetch(`${host}/preprocess/run?file_id=${this.seqId}`);
    if (!res.ok)
      return (this.message = `Error: Response not ok ${
        res.status
      } ${await res.text()}`);

    this.message = await res.text();
  }
}
</script>

<style>
#app {
  max-width: 600px;
  margin: auto;
  text-align: center;
}

#app * + * {
  margin-top: 20px;
}

.upload {
  height: 100px;
  width: 100%;
  border: 5px gray dashed;
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn {
  border: 1px gray solid;
  border-radius: 10px;
  padding: 10px;
}
</style>
