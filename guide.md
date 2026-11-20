# SkinAura UI and Asset Guide

This guide covers the three interfaces in this project: the Flask website, the Streamlit app, and the CustomTkinter desktop app. Make changes in the interface you actually run; the three layouts are separate, but their recommendations share data from `recommend.py`.

## Find the right file

| What you want to change | File |
| --- | --- |
| Flask page structure and visible text | `templates/index.html` |
| Flask colors, spacing, responsive layout, and backgrounds | `static/style.css` |
| Flask upload handling and result data | `app.py` |
| Streamlit page structure, widgets, and inline CSS | `web_app.py` |
| Desktop layout, widgets, colors, and image previews | `gui_app.py` |
| Product links, tips, profile tasks, and weather advice | `recommend.py` |
| Skin image size, dataset paths, and training options | `config.py` |

## Run the interface

Run these commands from the project directory in PowerShell or Command Prompt:

```powershell
python app.py
```

Open the local address printed by Flask. To run the other interfaces instead:

```powershell
streamlit run web_app.py
python gui_app.py
```

## Change the Flask website

### Edit the page layout

`templates/index.html` uses Jinja templates. The main upload and result columns are in `.workspace`; result sections use `.recommendation-grid`. Change the HTML when adding, removing, or reordering content. Keep form field names such as `photo`, `camera_data`, `skin_profile`, and `weather_condition` unchanged unless you also update `app.py`.

Use semantic headings and descriptive image alt text. For example, to add an image to a section:

```html
<img
  src="{{ url_for('static', filename='images/routine-example.webp') }}"
  alt="Example morning skincare routine"
>
```

The Flask route supplies result values to the template. If a new result field is added to `app.py`, render that field in `index.html` using Jinja syntax such as `{{ result.label }}` or a loop such as `{% for task in result.recommendation.morning_tasks %}`.

### Adjust spacing and responsive layout

Edit `static/style.css`. The desktop layout uses CSS Grid, and the mobile breakpoint is near the bottom of the file. For example:

```css
.workspace {
  grid-template-columns: 1fr 1.18fr;
}

@media (max-width: 780px) {
  .workspace {
    grid-template-columns: 1fr;
  }
}
```

The `[hidden]` rule is important: it keeps unselected upload and camera previews concealed even when another CSS rule sets `display`. Keep it in place so empty previews do not appear.

### Add an image or background

Put project-owned website images under `static/images/` (create the folder if needed). Reference them in HTML with `url_for` as shown above. From `static/style.css`, a relative URL is relative to the stylesheet, for example:

```css
.backdrop {
  background-image: url("images/skin-texture.webp");
  background-position: center;
  background-size: cover;
}
```

Use a relevant, clear image; keep a sensible file size, preserve aspect ratio, and use `object-fit: cover` for cropped previews. Use images you created or have permission to use. Add meaningful `alt` text to informative images; use empty alt text for purely decorative images.

## Change colors and visual style

### Flask

The main website palette is defined at the top of `static/style.css` in `:root`. Edit these variables to recolor the page consistently:

```css
:root {
  --ink: #18242a;
  --muted: #68777b;
  --green: #2d725d;
  --green-dark: #1d5145;
  --paper: #f4f5f0;
  --line: rgba(24, 36, 42, .12);
  --white: rgba(255, 255, 255, .9);
}
```

Use the variables in component rules, for example `color: var(--ink);`. Check text contrast after changing backgrounds, including button hover, warning, and sidebar states.

### Streamlit

`web_app.py` contains an inline `<style>` block near the top of the page. Its `:root` variables and selectors such as `.stApp` and `[data-testid="stSidebar"]` control the page and sidebar theme. Keep widget styling in that block near the other theme rules.

### Desktop

`gui_app.py` defines a `self.colors` dictionary in `SkinAuraApp.__init__`. Change its existing values and reuse them as `self.colors["paper"]`, `self.colors["ink"]`, and so on. Some small accents are direct hex values in widget definitions; update those too if they should follow the new palette.

## Add or change recommendation output

Edit `RECOMMENDATIONS` in `recommend.py`. Each profile supplies a summary, tips, two product names and URLs, hydration and diet notes, and separate morning and evening task lists. Keep the same keys for every profile because the interfaces render them directly.

Example product entry:

```python
"product_a": "Gentle Cleanser",
"product_a_url": "https://example.com/product",
```

Profile options are declared in `PROFILE_OPTIONS`. The image model only predicts classes present in its trained dataset (`dry`, `normal`, and `oily`). Combination and Sensitive are currently self-assessed guidance profiles, not image-model predictions. Do not add a new photo-predicted class just by adding a recommendation: you also need correctly labelled images for it, update `config.CLASS_NAMES`, prepare new data splits, retrain, and verify the saved class-label mapping.

## Change the desktop layout or add images

The desktop screen is assembled in `SkinAuraApp._build_ui()` in `gui_app.py`. It uses Tkinter's `grid` for the main page and `pack` for controls within panels. When adding a widget, choose one geometry manager per parent container; do not mix `grid` and `pack` in the same parent.

For a local image preview, the app already uses Pillow and `CTkImage` in `browse_image()`. Keep a reference to each `CTkImage` on the app instance (for example, `self.preview_image`) so Python does not release it while the widget is displaying it. Use a reasonable display size and `Image.open(path).convert("RGB")` for uploaded images.

## Change the Streamlit layout or add images

The page layout is written directly in `web_app.py` using Streamlit calls such as `st.columns`, `st.tabs`, `st.container`, and `st.image`. Put content in the tab or column where it belongs; add accompanying custom CSS to the inline style block near the top. For local images, build a path with `os.path.join` and pass it to `st.image`, or open it with Pillow when you need to resize or inspect it.

## Check your changes

After changing a page, run that interface and check both a wide and a narrow window. Confirm that images load, text fits, controls still submit, and hidden sections stay hidden. After changing recommendation data, check all profiles and confirm their morning/evening tasks and product links are present.

For photo-model changes, keep a held-out test split. The current bundled CNN is marked experimental; different ranked scores are not proof of reliable classification. Use diverse, correctly labelled images and evaluate model behavior before relying on it. SkinAura does not classify skin tone or diagnose skin conditions.
