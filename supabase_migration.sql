-- Run in the Supabase SQL Editor. Drops then recreates tables (order: messages before conversations).

drop table if exists od_messages;
drop table if exists od_conversations;

create table if not exists od_conversations (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now()
);

-- role = display name of the author (human name or agent name), not a role key
create table if not exists od_messages (
  id bigserial primary key,
  conversation_id uuid not null references od_conversations(id) on delete cascade,
  created_at timestamptz not null default now(),
  role text not null,
  message text not null
);

create index if not exists od_messages_conversation_id_idx on od_messages(conversation_id);
