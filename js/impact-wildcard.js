import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let refresh_btn = document.getElementById('comfy-refresh-button');

let orig = refresh_btn.onclick;

refresh_btn.onclick = function() {
	orig();
	api.fetchApi('/impact/wildcards/refresh');
};
