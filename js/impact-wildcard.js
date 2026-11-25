import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SOURCES = {
    LoraWildcardSource: { widget: "lora_wildcard", route: "/impact/get_items" },
    TextWildcardSource: { widget: "text_wildcard", route: "/impact/get_text_items" },
};

app.registerExtension({
    name: "Impact.LoraWildcardSource",
    nodeCreated(node) {
        const source = SOURCES[node.comfyClass];
        if (!source) return;
        const categoryWidget = node.widgets.find(w => w.name === "category");
        const wildcardWidget = node.widgets.find(w => w.name === source.widget);
        let requestId = 0;

        const updateItems = async (category, preserveValue) => {
            const currentRequest = ++requestId;
            try {
                const response = await api.fetchApi(`${source.route}?category=${encodeURIComponent(category)}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const items = await response.json();
                if (currentRequest !== requestId || categoryWidget.value !== category) return;
                wildcardWidget.options.values = items;
                if (!preserveValue) wildcardWidget.value = items[0] || "";
                node.setDirtyCanvas(true, true);
            } catch (error) {
                console.error("[Impact Wildcard] Cannot load category:", error);
            }
        };

        const callback = categoryWidget.callback;
        categoryWidget.callback = function (value) {
            callback?.apply(this, arguments);
            updateItems(value, false);
        };
        const configure = node.onConfigure;
        node.onConfigure = function () {
            const result = configure?.apply(this, arguments);
            updateItems(categoryWidget.value, true);
            return result;
        };
        updateItems(categoryWidget.value, true);
    }
});
