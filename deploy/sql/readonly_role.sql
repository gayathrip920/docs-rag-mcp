-- A login for the web API that can read the docs and write only the query log.
-- It cannot drop tables or change the docs, so a leaked address does far less harm.
--
-- Run this once, as the database owner. Replace CHANGE_ME with a long random password,
-- and keep that password out of git. Then point the API's DATABASE_URL at this role.

CREATE ROLE docs_api_reader WITH LOGIN PASSWORD 'CHANGE_ME';
GRANT USAGE ON SCHEMA public TO docs_api_reader;
GRANT SELECT ON pages, chunks TO docs_api_reader;
GRANT SELECT, INSERT ON queries TO docs_api_reader;
