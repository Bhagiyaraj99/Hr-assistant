"""
One-off script to generate a sample HR handbook PDF for local testing.
Not part of the production app — just test fixture data.
Run: python scripts/generate_sample_pdf.py
"""
from fpdf import FPDF

SECTIONS = [
    ("1. Vacation Policy",
     "Full-time employees accrue 15 paid vacation days per calendar year, "
     "accrued at a rate of 1.25 days per month. Vacation requests must be "
     "submitted to a manager at least 10 business days in advance for "
     "approval. Unused vacation days may be carried over into the next "
     "calendar year up to a maximum of 5 days; any balance beyond that is "
     "forfeited on December 31st. Part-time employees accrue vacation on a "
     "pro-rated basis according to their scheduled hours."),

    ("2. Sick Leave Policy",
     "Employees are entitled to 10 paid sick days per year, which do not "
     "carry over to the following year. Employees must notify their direct "
     "manager before the start of their shift if they are unable to work "
     "due to illness. A doctor's note is required for any sick leave "
     "exceeding 3 consecutive days. Sick leave may also be used to care for "
     "an immediate family member under the company's Family Care provision."),

    ("3. Remote Work Policy",
     "Employees may work remotely up to 3 days per week with manager "
     "approval, provided their role is designated as remote-eligible. "
     "Remote work arrangements must be reviewed and re-approved quarterly. "
     "Employees working remotely are expected to be available during core "
     "hours of 10 AM to 3 PM in their local time zone and must maintain a "
     "stable internet connection. The company will provide a one-time home "
     "office stipend of $300 for eligible remote employees."),

    ("4. Code of Conduct",
     "All employees are expected to act with integrity, respect, and "
     "professionalism in all workplace interactions, whether in person, "
     "virtual, or written communication. Harassment, discrimination, or "
     "retaliation of any kind will not be tolerated and should be reported "
     "immediately to HR or through the anonymous ethics hotline. Violations "
     "of the Code of Conduct may result in disciplinary action up to and "
     "including termination of employment."),

    ("5. Expense Reimbursement Policy",
     "Business-related expenses must be submitted through the expense "
     "portal within 30 days of the purchase date, accompanied by an "
     "itemized receipt. Expenses exceeding $500 require pre-approval from a "
     "department head before being incurred. Reimbursable categories "
     "include travel, client meals (up to $75 per person), and approved "
     "software subscriptions. Reimbursements are processed within 10 "
     "business days of approval."),

    ("6. Performance Review Process",
     "Formal performance reviews are conducted twice annually, in June and "
     "December. Reviews are based on goal completion, peer feedback, and "
     "manager assessment using the company's competency framework. "
     "Employees receive a written summary and a rating on a 5-point scale. "
     "Ratings of 4 or above are eligible for the annual merit increase "
     "cycle, subject to budget approval. New employees receive an informal "
     "90-day check-in in addition to the standard review cycle."),

    ("7. Termination and Resignation Policy",
     "Employees who wish to resign are asked to provide a minimum of 2 "
     "weeks' written notice to their manager and HR. The company reserves "
     "the right to terminate employment at any time, with or without "
     "cause, in accordance with applicable employment standards "
     "legislation. Final pay, including any accrued and unused vacation "
     "pay, will be issued within the timeframe required by provincial "
     "employment law. Exit interviews are conducted for all departing "
     "employees."),
]


def build_pdf(output_path: str = "data/sample_docs/employee_handbook.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 20, "Acme Corp Employee Handbook", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "Effective Date: January 2026", new_x="LMARGIN", new_y="NEXT", align="C")

    # One section per page so page-citation testing is easy to verify
    for title, body in SECTIONS:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 10, title)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, body)

    pdf.output(output_path)
    print(f"Sample PDF written to: {output_path}")


if __name__ == "__main__":
    build_pdf()