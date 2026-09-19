"""Three competing causal predictions; binary reports are not consciousness tests."""
import itertools


def predictions(donor_state,recipient_state,donor_code,recipient_code):
    bits=(donor_state,recipient_state,donor_code,recipient_code)
    if any(type(x) is not int or x not in (0,1) for x in bits):
        raise ValueError('Expected four binary integers')
    return dict(stateTransfer=donor_state^recipient_code,
                donorAnswer=donor_state^donor_code,
                recipientUnchanged=recipient_state^recipient_code)


def factorial():
    return [dict(donorState=d,recipientState=r,donorCode=dc,recipientCode=rc,
                 predictions=predictions(d,r,dc,rc))
            for d,r,dc,rc in itertools.product((0,1),repeat=4)]
