"""Four distinct predictions for cross-task interchange; no consciousness claim."""
TASKS = ('monitor', 'marker_first')


def predictions(donor_task, recipient_task, donor_state, recipient_state,
                donor_public, recipient_public, donor_code, recipient_code):
    if donor_task not in TASKS or recipient_task not in TASKS:
        raise ValueError('Unknown task')
    bits = (donor_state, recipient_state, donor_public, recipient_public, donor_code, recipient_code)
    if any(type(v) is not int or v not in (0, 1) for v in bits):
        raise ValueError('Expected six binary integers')
    donor_for_recipient = donor_state if recipient_task == 'monitor' else donor_public
    donor_for_donor = donor_state if donor_task == 'monitor' else donor_public
    recipient = recipient_state if recipient_task == 'monitor' else recipient_public
    return dict(recipientTaskContent=donor_for_recipient ^ recipient_code,
                donorBoolean=donor_for_donor ^ recipient_code,
                donorAnswer=donor_for_donor ^ donor_code,
                recipientUnchanged=recipient ^ recipient_code)
