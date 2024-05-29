class DateMismatch(Exception):
    def __init__(self, mismatch_files):
        self.mismatch_files = mismatch_files
        message = (
            "The following datasets does not have the same start and end date: "
            + ", ".join(mismatch_files)
        )
        super().__init__(message)
