def assign_relevance(final_result, date_unregistration, sum_click, click_75th):
    """
    Assign graded relevance label (0-3) to each student-course pair.

    date_unregistration values:
      '?'      -> student never withdrew (stayed enrolled)
      positive -> withdrew N days after course start
      negative -> withdrew before course started
      0        -> withdrew on day 0
    """
    try:
        unreg = None if str(date_unregistration).strip() == '?' \
                else float(date_unregistration)
    except (ValueError, TypeError):
        unreg = None

    if final_result in ['Distinction', 'Pass']:
        if sum_click >= click_75th:
            return 3
        return 2

    if final_result == 'Withdrawn':
        if unreg is not None and unreg <= 0:
            return 0
        if unreg is None or unreg > 60:
            return 1
        return 0

    return 0  # Fail or anything else