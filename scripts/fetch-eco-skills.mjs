// Fetches the third-party econometrics skill (MIT) that the engine bundles at runtime.
// It is not stored in this repo: it is cloned at a pinned commit so every checkout gets the same files.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = "https://github.com/xiaomihu1992/econometrics-skill.git";
const COMMIT = "f2cd97678e56cf8606a30f3e54dfe35365d31126";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "engine", "src", "qtrix_engine", "agents", "SKILLS", "eco-skills", "econometrics-skill");

if (existsSync(join(target, "lib"))) {
  console.log("eco-skills already present, skipping");
} else {
  mkdirSync(target, { recursive: true });
  const git = (...args) => execFileSync("git", args, { cwd: target, stdio: "inherit" });
  git("init", "-q");
  git("remote", "add", "origin", REPO);
  git("fetch", "-q", "--depth", "1", "origin", COMMIT);
  git("checkout", "-q", "FETCH_HEAD");
  console.log(`eco-skills fetched at ${COMMIT.slice(0, 7)}`);
}
