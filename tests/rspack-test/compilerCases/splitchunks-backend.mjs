import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Each probe gets its own process: backend selection must not change the native
// environment of another compiler case running concurrently in this worker.
const PROBE_SCRIPT = String.raw`
const fs = require("node:fs");
const path = require("node:path");
const { createHash } = require("node:crypto");

(async () => {
	const { rspack, experiments } = await import(process.argv[1]);
	const root = process.argv[2];
	const compiler = rspack({
		mode: "production",
		context: root,
		entry: { a: "./a.js", b: "./b.js", c: "./c.js", d: "./d.js" },
		target: "node",
		devtool: false,
		cache: false,
		output: {
			path: path.join(root, "dist"),
			filename: "[name].js",
			chunkFilename: "[name].js",
			uniqueName: "split-chunks-backend-test"
		},
		optimization: {
			minimize: false,
			concatenateModules: false,
			usedExports: false,
			moduleIds: "named",
			chunkIds: "named",
			splitChunks: {
				chunks: "all",
				dedupDepth: 1,
				minChunks: 2,
				minSize: 1,
				maxInitialRequests: Infinity,
				maxAsyncRequests: Infinity,
				cacheGroups: {
					default: false,
					defaultVendors: false,
					shared: { test: /[\\/]m[01]\.js$/ }
				}
			}
		},
		plugins: [new experiments.VirtualModulesPlugin({
			"a.js": 'import x from "./m0.js"; import y from "./m1.js"; console.log(x, y);',
			"b.js": 'import x from "./m0.js"; import y from "./m1.js"; console.log(y, x);',
			"c.js": 'import x from "./m0.js"; console.log(x);',
			"d.js": 'import y from "./m1.js"; console.log(y);',
			"m0.js": 'export default "module-zero";',
			"m1.js": 'export default "module-one";'
		})]
	});

	let result;
	try {
		const stats = await new Promise((resolve, reject) =>
			compiler.run((error, stats) => error ? reject(error) : resolve(stats))
		);
		if (stats.hasErrors()) {
			throw new Error(stats.toString({ all: false, errors: true, errorDetails: true }));
		}
		result = {
			assets: stats.toJson({ all: false, assets: true }).assets
				.map(({ name }) => [name, createHash("sha256")
					.update(fs.readFileSync(path.join(root, "dist", name))).digest("hex")])
				.sort(([a], [b]) => a.localeCompare(b))
		};
	} catch (error) {
		result = { error: String(error) };
	} finally {
		await new Promise((resolve, reject) =>
			compiler.close(error => error ? reject(error) : resolve())
		);
	}
	console.log(JSON.stringify(result));
})().catch(error => {
	console.error(error);
	process.exit(1);
});
`;

/** @type {import('@rspack/test-tools').TCompilerCaseConfig} */
export default {
	description: "selects SplitChunks backends without changing output or hiding CUDA failures",
	async run() {
		const root = fs.mkdtempSync(path.join(os.tmpdir(), "rspack-split-chunks-backend-"));
		const probe = (overrides = {}) => {
			const env = { ...process.env };
			delete env.RSPACK_SPLIT_CHUNKS_BACKEND;
			delete env.RSPACK_SPLIT_CHUNKS_TELEMETRY;
			delete env.RSPACK_SPLIT_CHUNKS_VERIFY;
			Object.assign(env, overrides);
			const raw = execFileSync(process.execPath, [
				"-e", PROBE_SCRIPT, import.meta.resolve("@rspack/core"), root
			], { env, encoding: "utf8", timeout: 20_000, windowsHide: true });
			return JSON.parse(raw.trim().split("\n").at(-1));
		};
		const records = (file) => fs.readFileSync(file, "utf8")
			.trim().split("\n").map(line => JSON.parse(line));
		try {
			const baseline = probe();
			expect(baseline.error).toBeUndefined();
			expect(baseline.assets.length).toBeGreaterThan(4);
			for (const backend of ["cpu", "auto"]) {
				const telemetry = path.join(root, `${backend}.jsonl`);
				const result = probe({
					RSPACK_SPLIT_CHUNKS_BACKEND: backend,
					RSPACK_SPLIT_CHUNKS_TELEMETRY: telemetry
				});
				expect(result).toEqual(baseline);
				const rounds = records(telemetry);
				expect(rounds.length).toBeGreaterThan(0);
				for (const round of rounds) {
					expect(round.schema_version).toBe(1);
					expect(round.event).toBe("split_chunks_intersection_round");
					expect(round.requested_backend).toBe(backend);
					expect(round.backend).toBe("cpu");
					expect(round.kernel_launches).toBe(0);
					expect(round.pair_count).toBeGreaterThan(0);
					expect(round.fallback_reason).toBe(backend === "auto"
						? "below-auto-word-ops-threshold" : null);
				}
			}
			expect(probe({ RSPACK_SPLIT_CHUNKS_BACKEND: "invalid" }).error)
				.toContain("RSPACK_SPLIT_CHUNKS_BACKEND must be cpu, cuda, or auto");
			expect(probe({ RSPACK_SPLIT_CHUNKS_TELEMETRY: "relative.jsonl" }).error)
				.toContain("RSPACK_SPLIT_CHUNKS_TELEMETRY must be an absolute JSONL file path");

			const telemetry = path.join(root, "cuda.jsonl");
			const forced = probe({
				RSPACK_SPLIT_CHUNKS_BACKEND: "cuda",
				RSPACK_SPLIT_CHUNKS_TELEMETRY: telemetry,
				RSPACK_SPLIT_CHUNKS_VERIFY: "1"
			});
			if (forced.error) {
				// Accept only a missing backend/device or an incompatible driver.
				// Kernel, allocation, and verification errors must fail this case.
				expect(forced.error).toMatch(
					/SplitChunks CUDA backend failed: (?:CUDA SplitChunks support is not compiled into this binding;|[^\n]*\(CUDA status (?:35|100|803|804)\))/
				);
			} else {
				expect(forced).toEqual(baseline);
				const rounds = records(telemetry);
				expect(rounds.reduce((sum, round) => sum + round.kernel_launches, 0))
					.toBeGreaterThan(0);
				for (const round of rounds) {
					expect(round.requested_backend).toBe("cuda");
					expect(round.backend).toBe("cuda");
					expect(round.fallback_reason).toBeNull();
					expect(round.verified).toBe(true);
				}
			}
		} finally {
			fs.rmSync(root, { recursive: true, force: true });
		}
	}
};
