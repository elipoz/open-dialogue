-- RLS (Row Level Security) for od_conversations and od_messages
-- Run in Supabase SQL Editor after the main tables exist.
--
-- Your app uses the service_role key (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY).
-- The service_role key bypasses RLS, so the app keeps working. We enable RLS and
-- revoke anon/authenticated access so only the app (service_role) can read/write.
-- Ensure your app never exposes the service_role key (server-side only).

-- 1) Enable RLS on both tables (clears the vulnerability alert)
ALTER TABLE "public"."od_conversations" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "public"."od_messages" ENABLE ROW LEVEL SECURITY;

-- 2) Revoke table privileges from anon and authenticated so they cannot access data
REVOKE ALL ON "public"."od_conversations" FROM anon, authenticated;
REVOKE ALL ON "public"."od_messages" FROM anon, authenticated;

-- Optional: if you later add Supabase Auth and owner_id/tenant_id columns,
-- add policies like the ones in the vulnerability alert and grant back as needed.
