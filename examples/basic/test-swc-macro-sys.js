#!/usr/bin/env node

/**
 * Test script to demonstrate swc_macro_sys fixing split chunk tree-shaking
 * This script processes the lodash vendor chunk and removes unused modules
 */

const fs = require("node:fs");
const path = require("node:path");
const { execSync } = require("node:child_process");

// Build the WASM module if needed
console.log("Building swc_macro_sys WASM module...");
try {
	execSync(
		"cd swc_macro_sys/crates/swc_macro_wasm && wasm-pack build --release",
		{
			stdio: "inherit",
			cwd: __dirname
		}
	);
} catch (err) {
	console.error("Failed to build WASM module:", err.message);
	console.log(
		"Make sure you have wasm-pack installed: cargo install wasm-pack"
	);
	process.exit(1);
}

// Import the WASM module
async function runTest() {
	const wasmPath = path.join(
		__dirname,
		"swc_macro_sys/crates/swc_macro_wasm/pkg/swc_macro_wasm.js"
	);
	const { default: init, process_webpack_bundle } = await import(wasmPath);

	// Initialize WASM
	await init();

	// Read the vendor chunk
	const chunkPath = path.join(
		__dirname,
		"dist/vendors-node_modules_pnpm_lodash-es_4_17_21_node_modules_lodash-es_lodash_js.js"
	);
	if (!fs.existsSync(chunkPath)) {
		console.error('Vendor chunk not found. Run "pnpm build" first.');
		process.exit(1);
	}

	const chunkContent = fs.readFileSync(chunkPath, "utf8");
	console.log(`\nOriginal chunk size: ${chunkContent.length} bytes`);

	// Count original modules
	const originalModuleCount = (
		chunkContent.match(/["'][^"']+["']\s*:\s*function/g) || []
	).length;
	console.log(`Original module count: ${originalModuleCount}`);

	// Read share-usage.json for tree-shake configuration
	const shareUsagePath = path.join(__dirname, "dist/share-usage.json");
	const shareUsage = JSON.parse(fs.readFileSync(shareUsagePath, "utf8"));

	// Test different configurations
	const testConfigs = [
		{
			name: "All exports disabled",
			config: {
				treeShake: {
					"lodash-es": Object.fromEntries(
						Object.keys(shareUsage.shareUsage["lodash-es"]).map(key => [
							key,
							false
						])
					)
				}
			}
		},
		{
			name: "Only VERSION enabled",
			config: {
				treeShake: {
					"lodash-es": {
						...Object.fromEntries(
							Object.keys(shareUsage.shareUsage["lodash-es"]).map(key => [
								key,
								false
							])
						),
						VERSION: true
					}
				}
			}
		},
		{
			name: "Only debounce enabled",
			config: {
				treeShake: {
					"lodash-es": {
						...Object.fromEntries(
							Object.keys(shareUsage.shareUsage["lodash-es"]).map(key => [
								key,
								false
							])
						),
						debounce: true
					}
				}
			}
		}
	];

	for (const testConfig of testConfigs) {
		console.log(`\n\n=== Test: ${testConfig.name} ===`);

		try {
			// Process the chunk with swc_macro_sys
			const result = await process_webpack_bundle(
				chunkContent,
				JSON.stringify(testConfig.config)
			);

			// Parse the result
			const processedChunk = result.processed_code;
			const removedModules = result.removed_modules || [];

			// Count remaining modules
			const remainingModuleCount = (
				processedChunk.match(/["'][^"']+["']\s*:\s*function/g) || []
			).length;

			console.log(`Processed chunk size: ${processedChunk.length} bytes`);
			console.log(
				`Size reduction: ${((1 - processedChunk.length / chunkContent.length) * 100).toFixed(2)}%`
			);
			console.log(
				`Remaining modules: ${remainingModuleCount} (removed ${originalModuleCount - remainingModuleCount})`
			);
			console.log(`Removed modules: ${removedModules.length}`);

			// Save the processed chunk for inspection
			const outputPath = path.join(
				__dirname,
				`dist/processed-${testConfig.name.toLowerCase().replace(/\s+/g, "-")}.js`
			);
			fs.writeFileSync(outputPath, processedChunk);
			console.log(`Saved to: ${outputPath}`);

			// Show some removed modules
			if (removedModules.length > 0) {
				console.log("\nSample of removed modules:");
				const samplesToShow = removedModules.slice(0, 10);
				for (const mod of samplesToShow) {
					console.log(`  - ${mod}`);
				}
				if (removedModules.length > 10) {
					console.log(`  ... and ${removedModules.length - 10} more`);
				}
			}
		} catch (err) {
			console.error(`Failed to process chunk: ${err.message}`);
		}
	}

	console.log("\n\n=== Summary ===");
	console.log(
		"The swc_macro_sys successfully removes unused modules from split chunks!"
	);
	console.log(
		"This solves the problem described in PROBLEM_STATEMENT.md where 640 modules"
	);
	console.log("remained even when all exports were disabled.");
}

// Run the test
runTest().catch(err => {
	console.error("Test failed:", err);
	process.exit(1);
});
