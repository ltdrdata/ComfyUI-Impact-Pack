import { app } from "../../scripts/app.js";

console.log("[Impact Wildcard] Extension script loaded and registering...");

app.registerExtension({
    name: "Impact.LoraWildcardSource",
    nodeCreated(node, app) {
        if (node.comfyClass === "LoraWildcardSource") {
            console.log("[Impact Wildcard] LoraWildcardSource node created in UI:", node);
            
            const categoryWidget = node.widgets.find(w => w.name === "category");
            const wildcardWidget = node.widgets.find(w => w.name === "lora_wildcard");
            
            console.log("[Impact Wildcard] Found widgets:", {
                categoryWidget: categoryWidget ? "YES" : "NO",
                wildcardWidget: wildcardWidget ? "YES" : "NO"
            });
            
            if (categoryWidget && wildcardWidget) {
                const updateItems = async (category, preserveValue = true) => {
                    console.log(`[Impact Wildcard] Fetching items for category: "${category}" (preserveValue: ${preserveValue})`);
                    try {
                        const response = await fetch(`/impact/get_items?category=${category}`);
                        if (response.ok) {
                            const items = await response.json();
                            console.log("[Impact Wildcard] Items successfully fetched:", items);
                            
                            wildcardWidget.options.values = items;
                            
                            if (preserveValue && items.includes(wildcardWidget.value)) {
                                console.log("[Impact Wildcard] Preserving current widget value:", wildcardWidget.value);
                            } else {
                                wildcardWidget.value = items[0] || "";
                                console.log("[Impact Wildcard] Value set to first option:", wildcardWidget.value);
                            }
                            node.setDirtyCanvas(true, true);
                            app.graph.setDirtyCanvas(true, true);
                        } else {
                            console.error("[Impact Wildcard] Server returned error response:", response.status, response.statusText);
                        }
                    } catch (e) {
                        console.error("[Impact Wildcard] Error loading category items:", e);
                    }
                };
                
                // Override the category change callback
                const originalCallback = categoryWidget.callback;
                categoryWidget.callback = function (value) {
                    console.log("[Impact Wildcard] Category widget changed callback fired. New value:", value);
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updateItems(value, false);
                };
                
                // Ensure synchronization when the workflow is configured (loaded from file)
                const originalOnConfigure = node.onConfigure;
                node.onConfigure = function (serialised_data) {
                    console.log("[Impact Wildcard] Node onConfigure fired. Category value:", categoryWidget.value);
                    if (originalOnConfigure) {
                        originalOnConfigure.apply(this, arguments);
                    }
                    updateItems(categoryWidget.value, true);
                };
                
                // Populate the list on node creation (preserves the initial workflow value)
                updateItems(categoryWidget.value, true);
            }
        }
    }
});
