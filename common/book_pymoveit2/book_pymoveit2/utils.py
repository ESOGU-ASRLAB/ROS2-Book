from typing import Type


def enum_to_str(enum_class: Type, value: int) -> str:
    """
    ROS 2 mesajındaki bir sayısal sabitin (enum) adını döndürür.

    Argümanlar:
        enum_class: Sabitleri içeren ROS 2 mesaj sınıfı.
        value: Sabitin tamsayı değeri.

    Dönüş:
        str: Sabitin adı; bulunamazsa değer ve "UNKNOWN NAME" ifadesi.
    """
    mapping = {}
    # Sınıftaki tüm öznitelikleri dolaş
    for attr_name in dir(enum_class):
        # Yalnızca büyük harfli öznitelikler (enum sabitleri için yaygın gelenek)
        if attr_name.isupper():
            attr_value = getattr(enum_class, attr_name)
            # Özniteliğin tamsayı olup olmadığına bak (enum değerleri genellikle int'tir)
            if isinstance(attr_value, int):
                mapping[attr_value] = attr_name
    return mapping.get(value, f"{value} :UNKNOWN NAME")
