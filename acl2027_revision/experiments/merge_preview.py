"""Builds paper/full_revised_preview.pdf = revised main body (pp. 1-12)
+ original Appendices A-Q (original pp. 13-100) + new Appendices R-X."""
from pypdf import PdfReader, PdfWriter
P = "../paper/"
w = PdfWriter()
main = PdfReader(P + "main.pdf")
orig = PdfReader(P + "original_submission.pdf")
new = PdfReader(P + "appendix_new.pdf")
assert len(main.pages) == 12, len(main.pages)
for pg in main.pages: w.add_page(pg)
for pg in orig.pages[12:100]: w.add_page(pg)
for pg in new.pages: w.add_page(pg)
w.write(P + "full_revised_preview.pdf")
print("pages:", len(w.pages))
