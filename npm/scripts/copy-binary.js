const fs = require("fs");
const path = require("path");
const os = require("os");

const scriptDir = __dirname;
const npmDir = path.dirname(scriptDir);
const whisperRoot = path.dirname(npmDir);

const platformKey = `${os.platform()}-${os.arch()}`;
const packageDir = path.join(npmDir, "packages", platformKey);

if (!fs.existsSync(packageDir)) {
  console.error(`Error: No package found for ${platformKey}`);
  console.error(`Expected: ${packageDir}`);
  process.exit(1);
}

// Possible source directories for the built addon
const possibleSourceDirs = [
  path.join(whisperRoot, "examples", "addon.node", "build", "Release"),
  path.join(whisperRoot, "build", "examples", "addon.node"),
];

const sourceDir = possibleSourceDirs.find((dir) =>
  fs.existsSync(path.join(dir, "addon.node.node"))
);

if (!sourceDir) {
  console.error("Error: Binary not found.");
  console.error("Tried:");
  for (const dir of possibleSourceDirs) {
    console.error(`  - ${path.join(dir, "addon.node.node")}`);
  }
  console.error("");
  console.error("Please build the addon first:");
  console.error("  cd examples/addon.node");
  console.error("  npm install");
  console.error("  npx cmake-js compile");
  process.exit(1);
}

// Copy the main addon binary
const addonSource = path.join(sourceDir, "addon.node.node");
const addonDest = path.join(packageDir, "whisper.node");

fs.copyFileSync(addonSource, addonDest);

const stats = fs.statSync(addonDest);
const sizeMb = (stats.size / (1024 * 1024)).toFixed(2);

console.log(`Copied binary to ${addonDest}`);
console.log(`Binary size: ${sizeMb} MB`);

// On Windows, also copy OpenVINO DLLs if present
if (os.platform() === "win32") {
  const openvinoDlls = [
    "openvino.dll",
    "openvino_intel_cpu_plugin.dll",
    "openvino_intel_gpu_plugin.dll",
    "openvino_auto_plugin.dll",
    "openvino_ir_frontend.dll",
    "tbb12.dll",
    "pugixml.dll",
  ];

  let copiedDlls = 0;
  let totalDllSize = 0;

  for (const dll of openvinoDlls) {
    const dllSource = path.join(sourceDir, dll);
    if (fs.existsSync(dllSource)) {
      const dllDest = path.join(packageDir, dll);
      fs.copyFileSync(dllSource, dllDest);
      const dllStats = fs.statSync(dllDest);
      totalDllSize += dllStats.size;
      copiedDlls++;
    }
  }

  if (copiedDlls > 0) {
    const dllSizeMb = (totalDllSize / (1024 * 1024)).toFixed(2);
    console.log(`Copied ${copiedDlls} OpenVINO DLLs (${dllSizeMb} MB)`);
  }
}
