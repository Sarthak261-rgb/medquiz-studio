# Data license

The code in this repository is MIT licensed (see `LICENSE`). The data is not.

Everything under `data/processed/`, `data/figures/`, `runs/`, `data/reviews/`,
`reports/` and `releases/` contains or is derived from:

> Betts, J. G., et al. *Anatomy and Physiology 2e*. OpenStax, Rice University.
> https://openstax.org/details/books/anatomy-and-physiology-2e
> Source files: https://github.com/openstax/osbooks-anatomy-physiology at commit
> `5ae32b3f4bc24ed003e91dc38bf47dba80751044`.

That includes the generated questions, the generated images in
`runs/images-v1/images/` and the LoRA weights in `runs/images-v1/lora/`, which were
trained on the book's own anatomical drawings.

The book is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
So everything derived from it here is released under the same license: you can
share and adapt it with attribution, not for commercial use, and adaptations must
carry the same license.

Changes from the original: the text was cleaned and split into passages, and
multiple-choice questions were generated from those passages by a language model,
checked automatically, and reviewed by a person. Figures credited to third parties
were left out of the image training set, and the rest were padded to 512 px. The
review questions in `data/processed/book_questions.jsonl` are the book's own,
copied unchanged.

Access the original book for free at openstax.org.

Every generated image is synthetic and says so, in a banner inside the picture and
in its EXIF metadata. None of them shows a real patient.

Nothing here has been reviewed by a clinician or a medical educator. Do not use the
questions or the images for teaching, assessment or any medical purpose without
that review.
