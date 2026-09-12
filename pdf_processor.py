import pymupdf
import os
import re


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_line(line):

    line = line.strip()

    line = re.sub(
        r"\s+",
        " ",
        line
    )

    return line


# ============================================================
# GET NON-EMPTY LINES
# ============================================================

def get_lines(text):

    lines = []

    for line in text.split("\n"):

        line = clean_line(line)

        if line:
            lines.append(line)

    return lines


# ============================================================
# CHECK FOR JOURNAL / HEADER LINE
# ============================================================

def is_header_line(line):

    lower = line.lower()

    header_patterns = [

        "volume",
        "vol.",
        "issue",
        "journal",
        "psychological review",
        "available online",
        "copyright",
        "elsevier",
        "springer"
    ]

    for pattern in header_patterns:

        if pattern in lower:
            return True

    return False


# ============================================================
# CHECK FOR AFFILIATION
# ============================================================

def is_affiliation(line):

    lower = line.lower()

    keywords = [

        "university",
        "college",
        "institute",
        "department",
        "laboratory",
        "laboratories",
        "school",
        "faculty",
        "academy",
        "hospital",
        "research center",
        "research centre",
        "corporation",
        "company"
    ]

    for keyword in keywords:

        if keyword in lower:
            return True

    return False


# ============================================================
# CHECK FOR CONTACT INFORMATION
# ============================================================

def is_contact(line):

    lower = line.lower()

    if "@" in line:
        return True

    if "e-mail" in lower:
        return True

    if "email" in lower:
        return True

    return False


# ============================================================
# CHECK FOR SECTION HEADING
# ============================================================

def is_section_heading(line):

    lower = line.lower().strip()

    headings = [

        "abstract",
        "introduction",
        "background",
        "keywords",
        "key words",
        "methodology",
        "methods",
        "results",
        "discussion",
        "conclusion"
    ]

    return lower in headings


# ============================================================
# CHECK FOR AUTHOR-LIKE LINE
# ============================================================

def is_author_like(line):

    if not line:
        return False

    if is_header_line(line):
        return False

    if is_affiliation(line):
        return False

    if is_contact(line):
        return False

    if is_section_heading(line):
        return False

    # Author names normally aren't very long
    if len(line) > 100:
        return False

    lower = line.lower()

    # Avoid obvious prose
    prose_starts = [

        "this ",
        "the ",
        "we ",
        "if ",
        "according ",
        "however ",
        "in this ",
        "when ",
        "although ",
        "because "
    ]

    for start in prose_starts:

        if lower.startswith(start):
            return False

    # Must contain letters
    if not re.search(
        r"[A-Za-zÀ-ÿ]",
        line
    ):
        return False

    # --------------------------------------------------------
    # Typical author formats
    # --------------------------------------------------------

    # Example:
    # F. ROSENBLATT
    # John Smith
    # John A. Smith
    # A. Kumar, B. Sharma
    # Smith J.
    # --------------------------------------------------------

    words = line.split()

    if len(words) > 15:
        return False

    # If the line is all uppercase, it can still be an author.
    # Example:
    # F. ROSENBLATT
    #
    # But reject long uppercase title-like lines.
    letters = [
        c for c in line
        if c.isalpha()
    ]

    uppercase_ratio = 0

    if letters:

        uppercase_ratio = (
            sum(
                c.isupper()
                for c in letters
            )
            /
            len(letters)
        )

    # Long all-uppercase lines are probably title
    if uppercase_ratio > 0.75 and len(line) > 35:
        return False

    # --------------------------------------------------------
    # Name-like structure
    # --------------------------------------------------------

    name_pattern = re.compile(
        r"""
        ^
        [A-Za-zÀ-ÿ'’.\-]+
        (?:
            [\s,;]+
            [A-Za-zÀ-ÿ'’.\-]+
        ){0,12}
        $
        """,
        re.VERBOSE
    )

    if name_pattern.match(line):

        return True

    return False


# ============================================================
# EXTRACT TITLE
# ============================================================

def extract_title_from_first_page(
    first_page_text
):

    lines = get_lines(
        first_page_text
    )

    if not lines:
        return None

    # --------------------------------------------------------
    # Find first strong title line
    # --------------------------------------------------------

    title_start = None

    for i, line in enumerate(lines):

        if i > 30:
            break

        if is_header_line(line):
            continue

        if is_affiliation(line):
            continue

        if is_author_like(line):
            continue

        if is_section_heading(line):
            continue

        # Title usually has several words
        if len(line.split()) >= 3:

            title_start = i
            break

    if title_start is None:
        return None

    # --------------------------------------------------------
    # Collect title lines
    # --------------------------------------------------------

    title_parts = []

    for i in range(
        title_start,
        min(
            title_start + 10,
            len(lines)
        )
    ):

        line = lines[i]

        # Stop when actual author is reached.
        #
        # IMPORTANT:
        # We only consider an author AFTER at least
        # one title line has already been collected.
        # ----------------------------------------------------

        if (
            len(title_parts) > 0
            and is_author_like(line)
        ):
            break

        if is_affiliation(line):
            break

        if is_contact(line):
            break

        if is_section_heading(line):
            break

        title_parts.append(line)

    if not title_parts:
        return None

    title = " ".join(
        title_parts
    )

    # Remove footnote numbers at end
    title = re.sub(
        r"\s*\d+\s*$",
        "",
        title
    )

    # Clean whitespace
    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


# ============================================================
# EXTRACT AUTHORS
# ============================================================

def extract_authors_from_first_page(
    first_page_text
):

    lines = get_lines(
        first_page_text
    )

    if not lines:
        return None

    # --------------------------------------------------------
    # First determine where title ends.
    # --------------------------------------------------------

    title_start = None

    for i, line in enumerate(lines):

        if i > 30:
            break

        if is_header_line(line):
            continue

        if is_affiliation(line):
            continue

        if is_section_heading(line):
            continue

        if len(line.split()) >= 3:

            title_start = i
            break

    if title_start is None:
        return None

    # --------------------------------------------------------
    # Find author AFTER title block.
    #
    # We do NOT simply stop at the first "author-like" line
    # because a title can itself look like a name.
    # --------------------------------------------------------

    title_end = None

    for i in range(
        title_start,
        min(
            title_start + 10,
            len(lines)
        )
    ):

        line = lines[i]

        # Affiliation means title has ended,
        # but author should have appeared before it.
        if is_affiliation(line):
            title_end = i
            break

        if is_contact(line):
            title_end = i
            break

        # Section means no author found
        if is_section_heading(line):
            title_end = i
            break

        # ----------------------------------------------------
        # We consider a line an author only if:
        #
        # 1. It is relatively short
        # 2. It follows at least one title line
        # 3. It looks like a name
        #
        # ----------------------------------------------------

        if (
            i > title_start
            and is_author_like(line)
        ):

            title_end = i
            break

    if title_end is None:

        title_end = title_start + 1

    # --------------------------------------------------------
    # Search for one or more authors.
    # --------------------------------------------------------

    authors = []

    for i in range(
        title_end,
        min(
            title_end + 12,
            len(lines)
        )
    ):

        line = lines[i]

        # Stop when affiliation begins
        if is_affiliation(line):
            break

        if is_contact(line):
            break

        if is_section_heading(line):
            break

        # Footnote marker
        if re.fullmatch(
            r"[\d,*†‡]+",
            line
        ):
            continue

        if is_author_like(line):

            authors.append(line)

            continue

        # If we already found an author and then
        # encounter something that doesn't look like
        # another author, stop.
        if authors:
            break

    if not authors:
        return None

    # Remove duplicates
    unique_authors = []

    for author in authors:

        if author not in unique_authors:

            unique_authors.append(
                author
            )

    return ", ".join(
        unique_authors
    )


# ============================================================
# VALIDATE PDF METADATA AUTHOR
# ============================================================

def valid_metadata_author(
    author
):

    if not author:
        return False

    author = author.strip()

    if len(author) < 2:
        return False

    invalid = [

        "unknown",
        "anonymous",
        "author",
        "authors",
        "none",
        "null"
    ]

    if author.lower() in invalid:
        return False

    # Reject obvious journal metadata
    if "journal" in author.lower():
        return False

    if "review" in author.lower():
        return False

    if "volume" in author.lower():
        return False

    if "issue" in author.lower():
        return False

    return True


# ============================================================
# EXTRACT METADATA
# ============================================================

def extract_metadata(
    first_page_text,
    pdf_metadata,
    pdf_path
):

    # --------------------------------------------------------
    # FIRST: visible paper content
    # --------------------------------------------------------

    extracted_title = (
        extract_title_from_first_page(
            first_page_text
        )
    )

    extracted_author = (
        extract_authors_from_first_page(
            first_page_text
        )
    )

    # --------------------------------------------------------
    # SECOND: PDF metadata
    # --------------------------------------------------------

    metadata_title = pdf_metadata.get(
        "title"
    )

    metadata_author = pdf_metadata.get(
        "author"
    )

    if metadata_title:
        metadata_title = metadata_title.strip()

    if metadata_author:
        metadata_author = metadata_author.strip()

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if extracted_title:

        title = extracted_title

    elif metadata_title:

        title = metadata_title

    else:

        title = os.path.splitext(
            os.path.basename(
                pdf_path
            )
        )[0]

    # --------------------------------------------------------
    # AUTHOR
    # --------------------------------------------------------

    if extracted_author:

        author = extracted_author

    elif valid_metadata_author(
        metadata_author
    ):

        author = metadata_author

    else:

        author = "Unknown"

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    author = re.sub(
        r"\s+",
        " ",
        author
    ).strip()

    return title, author


# ============================================================
# EXTRACT COMPLETE PDF
# ============================================================

def extract_text_from_pdf(
    pdf_path
):

    document = pymupdf.open(
        pdf_path
    )

    pdf_metadata = document.metadata

    pages = []

    # --------------------------------------------------------
    # First page
    # --------------------------------------------------------

    first_page_text = ""

    if len(document) > 0:

        first_page_text = (
            document[0].get_text()
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    title, author = extract_metadata(
        first_page_text,
        pdf_metadata,
        pdf_path
    )

    # --------------------------------------------------------
    # Every page
    # --------------------------------------------------------

    for page_number, page in enumerate(
        document
    ):

        page_text = page.get_text()

        page_metadata = {

            "title": title,

            "author": author,

            "source": os.path.basename(
                pdf_path
            ),

            "page": page_number + 1
        }

        pages.append({

            "text": page_text,

            "metadata": page_metadata
        })

    document.close()

    return pages


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = "papers/paper2.pdf"

    pages = extract_text_from_pdf(
        pdf_path
    )

    print(
        f"Total pages: {len(pages)}"
    )

    print(
        "\n===== PAPER METADATA ====="
    )

    print(
        pages[0]["metadata"]
    )

    print(
        "\n===== FIRST PAGE ====="
    )

    print(
        pages[0]["text"][:3000]
    )