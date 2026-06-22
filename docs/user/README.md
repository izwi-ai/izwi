# Izwi Mintlify Docs

This directory is the Mintlify app root for the public Izwi user docs.

## Mintlify Setup

- Configure the Mintlify project as a monorepo.
- Set the documentation path to `/docs/user`.
- Run local Mintlify commands from this directory.

## Vercel Setup

The repository root includes `vercel.ts` rewrites for serving Mintlify at the
main-site `/docs` path.

Set one of these Vercel environment variables:

- `MINTLIFY_DOCS_ORIGIN`, for example `https://your-subdomain.mintlify.dev`
- `MINTLIFY_SUBDOMAIN`, for example `your-subdomain`

Then requests to `/docs` and `/docs/:match*` proxy to the matching Mintlify
`/docs` routes.
