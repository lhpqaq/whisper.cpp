import { platform, arch } from "os";
import { join } from "path";
import { existsSync } from "fs";
import type { WhisperAddon } from "./types";

/**
 * Supported platform-arch combinations
 */
const SUPPORTED_PLATFORMS: Record<string, string> = {
  "darwin-arm64": "@whisper-cpp-node/darwin-arm64",
  "win32-x64": "@whisper-cpp-node/win32-x64",
  // Future: add more platforms
  // "darwin-x64": "@whisper-cpp-node/darwin-x64",
  // "linux-x64": "@whisper-cpp-node/linux-x64",
};

/**
 * Get the platform key for current system
 */
function getPlatformKey(): string {
  return `${platform()}-${arch()}`;
}

/**
 * Get the platform-specific package name
 */
function getPlatformPackage(): string {
  const platformKey = getPlatformKey();
  const packageName = SUPPORTED_PLATFORMS[platformKey];

  if (!packageName) {
    const supported = Object.keys(SUPPORTED_PLATFORMS).join(", ");
    throw new Error(
      `Unsupported platform: ${platformKey}. ` +
        `Supported platforms: ${supported}`
    );
  }

  return packageName;
}

/**
 * Try to find the binary in workspace development paths
 */
function tryWorkspacePath(): string | null {
  const platformKey = getPlatformKey();

  // In monorepo development, the binary is in sibling package
  const possiblePaths = [
    // From dist/ folder: ../darwin-arm64/whisper.node
    join(__dirname, "..", "..", platformKey, "whisper.node"),
    // From src/ folder during ts-node: ../../darwin-arm64/whisper.node
    join(__dirname, "..", "..", "..", platformKey, "whisper.node"),
  ];

  for (const p of possiblePaths) {
    if (existsSync(p)) {
      return p;
    }
  }

  return null;
}

/**
 * Load the native addon for the current platform
 */
export function loadNativeAddon(): WhisperAddon {
  const packageName = getPlatformPackage();

  // First, try workspace development path
  const workspacePath = tryWorkspacePath();
  if (workspacePath) {
    return require(workspacePath) as WhisperAddon;
  }

  // Then try the installed package
  try {
    const binaryPath = require.resolve(join(packageName, "whisper.node"));
    return require(binaryPath) as WhisperAddon;
  } catch (error) {
    const err = error as NodeJS.ErrnoException;

    if (err.code === "MODULE_NOT_FOUND") {
      throw new Error(
        `Native binary not found. Please ensure ${packageName} is installed.\n` +
          `Try running: npm install ${packageName}\n` +
          `Original error: ${err.message}`
      );
    }

    throw new Error(
      `Failed to load native addon from ${packageName}: ${err.message}`
    );
  }
}
