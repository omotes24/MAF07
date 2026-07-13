from audit_image_proxy_fidelity import FROZEN_LEGACY_MACRO_AUROC


def test_frozen_legacy_order_is_explicit() -> None:
    ordered = sorted(FROZEN_LEGACY_MACRO_AUROC, key=FROZEN_LEGACY_MACRO_AUROC.get, reverse=True)
    assert ordered == ["PULSE", "DeepPrototype", "RC-MSPS", "MSPS", "CauchyStageTail"]
