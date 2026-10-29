"""Create, connect, edit, and run wildcard nodes through the Chrome UI.

The server's custom_wildcards must include tests/wildcards/samples.
No models are required. See docs/E2E_TEST_STRATEGY.md for server setup.
"""
from pathlib import Path

from playwright.sync_api import expect

from e2e_ui import browser_session, options


def main():
    args = options(__doc__)
    samples = Path(__file__).parent / "wildcards" / "samples" / "아름다운색.txt"
    colors = set(samples.read_text(encoding="utf-8").splitlines())
    with browser_session(args) as ui:
        page = ui.page
        ui.new_workflow()
        ui.add_node("ImpactWildcardProcessor", "ImpactWildcardProcessor", 350, 250)
        wildcard = page.get_by_placeholder("Wildcard Prompt (User input)", exact=True)
        populated = page.get_by_placeholder("Populated Prompt (Will be generated automatically)", exact=True)
        expect(populated).to_be_disabled()
        wildcard.fill("🌸 __아름다운색__")
        preview_id = ui.add_node("Preview as Text", "PreviewAny", 1050, 400)
        ui.connect("ImpactWildcardProcessor", "PreviewAny")
        expect(page.locator("textarea")).to_have_count(3)
        preview = page.locator("textarea").nth(2)

        def check_output(entry, expected):
            result = entry["outputs"][preview_id]["text"][0]
            assert result in expected, f"Unexpected expanded text: {result!r}"
            expect(populated).to_have_value(result)
            expect(preview).to_have_value(result)

        check_output(ui.run(), {f"🌸 {color}" for color in colors})
        expect(populated).to_be_disabled()

        ui.choose("ImpactWildcardProcessor", "mode", "fixed")
        expect(populated).to_be_enabled()
        populated.fill("고정 🌸")
        check_output(ui.run(), {"고정 🌸"})
        expect(populated).to_be_enabled()

        ui.choose("ImpactWildcardProcessor", "mode", "reproduce")
        populated.fill("재현 🌺")
        check_output(ui.run(), {"재현 🌺"})
        expect(populated).to_be_disabled()

        # The reproduce run must restore populate mode via websocket feedback.
        check_output(ui.run(), {f"🌸 {color}" for color in colors})
        print("PASS: UI node creation/linking, populate/fixed/reproduce, UTF-8, and preview updates")


if __name__ == "__main__":
    main()
