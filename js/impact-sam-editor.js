import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createSamTool } from "./sam-mask-editor.js";

if (app.extensionManager?.maskEditor) {
    app.registerExtension({
        name: "Comfy.Impact.SAMEditor",
        setup() {
            const result = app.extensionManager.maskEditor.registerTool(createSamTool(api));
            if (!result.ok) console.error(`[Impact Pack] SAM tool registration failed: ${result.reason}`);
        }
    });
} else {
    await import("./legacy-sam-editor.mjs");
}
