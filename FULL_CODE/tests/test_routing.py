from routing import get_department_for_issue


def test_payment_failure_routes_to_digital():
    assert get_department_for_issue("App Payment Failure") == "Digital"


def test_staff_misconduct_routes_to_hr():
    assert get_department_for_issue("Staff Misconduct") == "HR"
