# Presentation exports

The story presentation (milestone 24) is a Claude Slides deck, "Can You Predict How
Long People Live?", built from the seven notebooks: 16 slides with speaker notes (the
speaker notes double as the interview walkthrough script). Its charts are drawn by
`python -m site_build.presentation_charts` into `docs/presentation_charts/` from the
same saved results as everything else.

Export it from the deck's **Share → Export** menu and save the files here. The
export's own names are fine ("Can You Predict How Long People Live.pdf"): the build
picks the newest PDF, PowerPoint and HTML file in this folder and publishes them as:

| Export | Published as |
|---|---|
| PDF | `life-expectancy-story.pdf` |
| PowerPoint | `life-expectancy-story.pptx` |
| HTML | `life-expectancy-story.html` |

The export menu has no image option, so `build_site` makes the slide images itself:
it draws each PDF page as a 1920-pixel PNG, zips them as
`life-expectancy-story-slides.zip`, and saves the first as `cover.png`, the page's
thumbnail. It redoes this whenever the PDF is newer than the zip. It needs PyMuPDF
(`pymupdf` in `requirements.txt`).

Then run `python -m site_build.build_site`. It copies whichever files exist to
`scott/life-expectancy/presentation/` and builds the "The story, as a presentation"
section with View, PDF, PowerPoint, Web page and Slide images buttons. With no
exports here, the section is left out. After a retrain, run
`presentation_charts`, update the deck's images and numbers, re-export and rebuild.
