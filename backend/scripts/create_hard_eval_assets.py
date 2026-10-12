import json
from pathlib import Path
from textwrap import dedent


CORPUS_DIR = Path(
    "evaluation/hard_knowledge_base"
)

CASES_PATH = Path(
    "evaluation/hard_retrieval_cases.json"
)


DOCUMENTS = {
    "access_control_policy.txt": """
        KnowledgeOps AI Access Control Policy

        Access to production systems follows the principle of
        least privilege. Employees must receive only the access
        required for their assigned responsibilities.

        Standard production access requires approval from both
        the employee's direct manager and the Security Team.
        Database administrator privileges require an additional
        review by the infrastructure owner.

        Emergency production access may be granted during a
        critical incident when normal approval would cause an
        unacceptable operational delay. Emergency access expires
        automatically after four hours unless it is explicitly
        renewed by the Security Team.

        Privileged accounts must use multi-factor authentication.
        Privileged access is reviewed every 30 days. Accounts
        that no longer have a valid business justification must
        be disabled.

        Employees must never share production credentials.
        Access logs are retained for security review and may be
        examined during incident investigations.

        Requests for permanent access must include the system
        name, requested role, business justification, and the
        expected duration of access.
    """,

    "remote_work_policy.txt": """
        KnowledgeOps AI Remote Work Policy

        Employees may normally work remotely for up to three
        days per week. Working remotely for more than three days
        in the same week requires approval from the employee's
        direct manager.

        Company-managed devices must be used for access to
        internal systems. Employees working outside a company
        office must connect through the corporate VPN before
        accessing internal applications or confidential data.

        Public Wi-Fi should not be used for confidential work
        unless the employee is connected through the corporate
        VPN. Devices must remain protected by screen locking and
        disk encryption.

        Working temporarily from another country requires both
        manager and Human Resources approval. The request should
        normally be submitted at least ten business days before
        travel begins.

        Remote workers remain responsible for attending required
        meetings and maintaining reasonable availability during
        their agreed working hours.

        Approval for remote work does not automatically grant
        additional access to production systems. Production
        access remains governed by the separate Access Control
        Policy.
    """,

    "expense_travel_policy.txt": """
        KnowledgeOps AI Expense and Business Travel Policy

        Employees may claim reasonable expenses incurred for
        legitimate company business. A receipt is required for
        any individual expense above USD 25.

        Expense reports must normally be submitted within
        30 calendar days after the expense was incurred.
        Reports submitted after this period may require an
        explanation and additional Finance Team review.

        Domestic travel requires direct manager approval before
        booking when the expected cost exceeds the department's
        approved travel budget. International business travel
        requires both manager and Finance Team approval before
        booking.

        Personal expenses are not reimbursable. Employees must
        separate personal charges from company expenses whenever
        possible.

        Hotel, transportation, and meal expenses should be
        reasonable for the destination. Upgrades for personal
        preference are not reimbursable unless there is a
        documented business requirement.

        Exceptions to this policy require Finance Team approval.
        Approval of an expense does not override security or
        access-control requirements for company systems.
    """,

    "leave_attendance_policy.txt": """
        KnowledgeOps AI Leave and Attendance Policy

        Employees should plan annual leave with sufficient notice
        to reduce disruption to their teams.

        Requests for three or more consecutive days of annual
        leave must normally be submitted at least five business
        days before the first day of leave.

        Annual leave requires approval from the employee's direct
        manager. Approval depends on staffing needs, project
        commitments, and previously approved leave within the
        team.

        Employees who are unexpectedly unable to work because of
        illness should notify their manager before the scheduled
        start of the workday whenever reasonably possible.
        Medical documentation may be requested for extended sick
        leave where permitted by applicable rules.

        Unpaid leave requires review by Human Resources in
        addition to manager approval.

        Emergency leave should be reported as soon as reasonably
        possible. Employees are not expected to delay urgent
        personal matters solely to complete the normal advance
        request process.

        Leave approval does not change an employee's system
        access permissions or remote-work authorization.
    """,

    "incident_response_policy.txt": """
        KnowledgeOps AI Incident Response Policy

        Security and operational incidents are classified by
        severity. Priority 1 incidents represent the highest
        level of operational impact.

        A Priority 1 incident must be acknowledged within
        15 minutes after it is reported. The designated Incident
        Commander coordinates the response, assigns owners, and
        maintains the incident timeline.

        Engineers may request emergency production access during
        a critical incident. Such access remains subject to the
        emergency rules in the Access Control Policy and must not
        be treated as permanent authorization.

        Suspected exposure of personal or confidential data must
        be escalated to the Security Team. A suspected personal
        data breach must also be reported to the Privacy Officer
        within one hour of confirmation.

        Incident communications should distinguish verified facts
        from assumptions. Major decisions and remediation actions
        must be recorded.

        A post-incident review for a Priority 1 incident must be
        completed within five business days. The review should
        identify contributing factors, corrective actions, and
        owners for follow-up work.
    """,

    "vendor_contractor_policy.txt": """
        KnowledgeOps AI Vendor and Contractor Access Policy

        Contractors and vendors receive access only when required
        for an approved business purpose. A company employee must
        act as the sponsor for every external worker who receives
        access to company systems.

        Contractor access to production systems requires approval
        from both the internal sponsor and the Security Team.
        Temporary vendor production access normally expires after
        eight hours and must be renewed if additional time is
        required.

        External workers must use individually assigned accounts.
        Shared vendor credentials are prohibited. Multi-factor
        authentication is required for privileged external access.

        Before confidential company information is shared with a
        vendor, the responsible team must confirm that an
        appropriate data-processing or confidentiality agreement
        is in place and that the required security review has
        been completed.

        The internal sponsor is responsible for requesting access
        removal when the engagement ends. Security may revoke
        access earlier if suspicious activity is detected.

        Vendor approval does not automatically authorize access
        to employee records or other restricted information.
    """,

    "password_identity_policy.txt": """
        KnowledgeOps AI Password and Identity Policy

        User passwords must contain at least 14 characters.
        Employees should use the approved password manager to
        generate and store unique credentials.

        Multi-factor authentication is required for administrator
        accounts and other privileged identities. Teams may also
        require multi-factor authentication for additional
        systems based on risk.

        Passwords and authentication secrets must never be shared
        through email, chat messages, tickets, or documentation.
        Suspected credential exposure must be reported to the
        Security Team immediately.

        Service accounts must have a documented owner and business
        purpose. Service-account permissions must be reviewed at
        least every 90 days.

        Password resets require identity verification through the
        approved account-recovery process. Support personnel must
        not bypass identity verification simply because a request
        appears urgent.

        This policy defines authentication requirements but does
        not itself grant access to production systems. Production
        authorization remains governed by the Access Control
        Policy.
    """,

    "data_handling_policy.txt": """
        KnowledgeOps AI Data Handling Policy

        Company information is classified according to its
        sensitivity. Confidential information includes customer
        records, employee information, security data, and
        non-public business information.

        Confidential information must be encrypted while in
        transit and when stored on systems that support approved
        encryption controls.

        External sharing of confidential information requires a
        valid business purpose and approval from the responsible
        data owner. Security review may also be required when data
        is shared through a new vendor or technical platform.

        Employees must not copy confidential information into
        personal cloud-storage accounts or unapproved external
        services.

        Data should be retained only for the period required by
        the applicable retention schedule, contractual
        requirement, or legal obligation. When the retention
        period expires, information should be deleted or archived
        according to the approved process.

        Suspected unauthorized disclosure must be reported to the
        Security Team. Personal-data incidents may also require
        escalation under the Incident Response Policy.
    """,
}


CASES = [
    {
        "query": (
            "Who must approve standard employee access "
            "to production systems?"
        ),
        "expected_source": "access_control_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "How long does emergency production access "
            "last before it expires?"
        ),
        "expected_source": "access_control_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "How often are privileged access permissions "
            "reviewed?"
        ),
        "expected_source": "access_control_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "Can an employee work from home four days "
            "in one week without approval?"
        ),
        "expected_source": "remote_work_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "What connection should remote employees use "
            "before opening internal applications?"
        ),
        "expected_source": "remote_work_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "What approvals are needed to temporarily "
            "work from another country?"
        ),
        "expected_source": "remote_work_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "At what expense amount does a receipt become "
            "mandatory?"
        ),
        "expected_source": "expense_travel_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "What is the normal deadline for filing an "
            "expense report?"
        ),
        "expected_source": "expense_travel_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "Who must authorize international business "
            "travel before booking?"
        ),
        "expected_source": "expense_travel_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "How much advance notice is normally needed "
            "for three consecutive annual leave days?"
        ),
        "expected_source": "leave_attendance_policy.txt",
        "answerable": True,
    },
    {
        "query": "Who approves normal annual leave?",
        "expected_source": "leave_attendance_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "What should an employee do if illness "
            "unexpectedly prevents them from working?"
        ),
        "expected_source": "leave_attendance_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "What is the acknowledgement target for a "
            "Priority 1 incident?"
        ),
        "expected_source": "incident_response_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "Who coordinates teams during the highest "
            "severity incident?"
        ),
        "expected_source": "incident_response_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "Who must be informed when a personal data "
            "breach is confirmed?"
        ),
        "expected_source": "incident_response_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "What approvals are required before a "
            "contractor can access production?"
        ),
        "expected_source": "vendor_contractor_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "How long does temporary vendor production "
            "access normally remain valid?"
        ),
        "expected_source": "vendor_contractor_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "What must be checked before confidential "
            "information is sent to a vendor?"
        ),
        "expected_source": "vendor_contractor_policy.txt",
        "answerable": True,
    },

    {
        "query": "How many characters must a password contain?",
        "expected_source": "password_identity_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "Which identities are explicitly required "
            "to use multi-factor authentication?"
        ),
        "expected_source": "password_identity_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "How frequently must service-account "
            "permissions be reviewed?"
        ),
        "expected_source": "password_identity_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "How should confidential information be "
            "protected while being transmitted?"
        ),
        "expected_source": "data_handling_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "Who approves sharing confidential company "
            "information externally?"
        ),
        "expected_source": "data_handling_policy.txt",
        "answerable": True,
    },
    {
        "query": (
            "Can employees store confidential files in "
            "their personal cloud accounts?"
        ),
        "expected_source": "data_handling_policy.txt",
        "answerable": True,
    },

    {
        "query": (
            "What is the annual dental insurance "
            "reimbursement limit?"
        ),
        "expected_source": None,
        "answerable": False,
    },
    {
        "query": (
            "What time does the employee cafeteria "
            "open for breakfast?"
        ),
        "expected_source": None,
        "answerable": False,
    },
    {
        "query": (
            "How many weeks of paid parental leave "
            "does the company provide?"
        ),
        "expected_source": None,
        "answerable": False,
    },
    {
        "query": (
            "How often are employee laptops replaced "
            "with new hardware?"
        ),
        "expected_source": None,
        "answerable": False,
    },
    {
        "query": (
            "What percentage commission do sales "
            "employees earn?"
        ),
        "expected_source": None,
        "answerable": False,
    },
    {
        "query": (
            "Does the company reimburse monthly "
            "office parking fees?"
        ),
        "expected_source": None,
        "answerable": False,
    },
]


def main() -> None:
    CORPUS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, content in DOCUMENTS.items():
        path = CORPUS_DIR / filename

        path.write_text(
            dedent(content).strip() + "\n",
            encoding="utf-8",
        )

    CASES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CASES_PATH.write_text(
        json.dumps(
            CASES,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    answerable = sum(
        case["answerable"]
        for case in CASES
    )

    unanswerable = (
        len(CASES) - answerable
    )

    print(
        f"Documents created: {len(DOCUMENTS)}"
    )
    print(
        f"Evaluation cases: {len(CASES)}"
    )
    print(
        f"Answerable: {answerable}"
    )
    print(
        f"Unanswerable: {unanswerable}"
    )


if __name__ == "__main__":
    main()