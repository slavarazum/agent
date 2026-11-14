---
"@convex-dev/agent": patch
---

Store each message's text once instead of twice, so pages, prompt context, saves
and search read less. This is one-way: messages saved by this version keep their
text only in `text`, so an older version of the component fails schema
validation on them and can't show their text. `listMessages` and
`listUIMessages` take `textOnly`. Saved messages have no empty `sources` or
`warnings`.
