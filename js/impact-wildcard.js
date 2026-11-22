import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SOURCES = {
    LoraWildcardSource: { widget: "lora_wildcard", route: "/impact/get_items" },
    TextWildcardSource: { widget: "text_wildcard", route: "/impact/get_text_items", seedPerItem: true },
};

app.registerExtension({
    name: "Impact.LoraWildcardSource",
    nodeCreated(node) {
        const source = SOURCES[node.comfyClass];
        if (!source) return;
        const categoryWidget = node.widgets.find(w => w.name === "category");
        const wildcardWidget = node.widgets.find(w => w.name === source.widget);
        const seedWidget = source.seedPerItem ? node.widgets.find(w => w.name === "seed") : null;
        let requestId = 0;

        // TextWildcardSource: the seed is the line index, so keep it within the
        // line count. ComfyUI's "randomize" draws floor(random * (max - min)), so
        // max = count yields 0..count-1; "increment" stops at max, so wrap it to 0.
        const limitSeed = (count) => {
            if (!seedWidget || count < 1) return;
            seedWidget.options.max = count;
            if (seedWidget.value >= count) seedWidget.value = seedWidget.value % count;
        };
        if (seedWidget) {
            const seedCallback = seedWidget.callback;
            seedWidget.callback = function (value) {
                const count = wildcardWidget.options.values?.length || 0;
                if (count > 0 && seedWidget.value >= count) seedWidget.value = seedWidget.value % count;
                return seedCallback?.apply(this, [seedWidget.value, ...Array.prototype.slice.call(arguments, 1)]);
            };
        }

        const updateItems = async (category, preserveValue) => {
            const currentRequest = ++requestId;
            try {
                const response = await api.fetchApi(`${source.route}?category=${encodeURIComponent(category)}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const items = await response.json();
                if (currentRequest !== requestId || categoryWidget.value !== category) return;
                wildcardWidget.options.values = items;
                if (!preserveValue) wildcardWidget.value = items[0] || "";
                limitSeed(items.length);
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
