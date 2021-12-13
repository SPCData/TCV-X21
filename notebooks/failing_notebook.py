# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: tcv-x21
#     language: python
#     name: tcv-x21
# ---

# %% [markdown]
# # Dummy notebook for testing
#
# We want to make sure that our notebooks can still run after making changes to `tcv-x21`, so add them to a CI/CD step. To make sure that this testing is working, we also need a notebook which doesn't work, like this one.

# %%
print("Hello. I'm going to crash now.")

# %%
assert True == False, f"I did a stupid error"

# %%
print("If you're seeing this message, something has gone very wrong.")
