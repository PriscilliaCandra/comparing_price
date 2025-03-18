import streamlit as st

st.title("Saved Products")

if "saved_products" not in st.session_state or not st.session_state.saved_products:
    st.write("There are no saved products")
else:
    for product in st.session_state.saved_products:
        st.write(f"{product['name']}")
        st.write(f"Price: Rp {product['price']:,}")
        st.write(f"[See product detail]({product['link']})")
        st.write("---")
