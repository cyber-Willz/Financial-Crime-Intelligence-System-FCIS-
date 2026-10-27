-- PostgreSQL schema mirroring the engine's data model, for teams that persist the
-- engine's state relationally (e.g. via SeaORM). The Rust engine itself serialises to
-- JSON; these tables are the relational equivalent.

CREATE TABLE party (
    id                       BIGINT PRIMARY KEY,
    kind                     TEXT NOT NULL CHECK (kind IN ('individual','entity')),
    name                     TEXT NOT NULL CHECK (btrim(name) <> ''),
    aliases                  TEXT[] NOT NULL DEFAULT '{}',
    dob                      JSONB NOT NULL DEFAULT '[]',          -- [{year,month,day}]
    nationalities            TEXT[] NOT NULL DEFAULT '{}',         -- ISO alpha-2
    countries                TEXT[] NOT NULL DEFAULT '{}',
    identifiers              JSONB NOT NULL DEFAULT '[]',
    pep                      TEXT NOT NULL DEFAULT 'no' CHECK (pep IN ('no','pep','family_or_associate')),
    industry                 TEXT,
    channel                  TEXT NOT NULL DEFAULT 'face_to_face' CHECK (channel IN ('face_to_face','non_face_to_face','intermediary')),
    adverse_media            BOOLEAN NOT NULL DEFAULT FALSE,
    nominee_or_bearer_shares BOOLEAN NOT NULL DEFAULT FALSE,
    status                   TEXT NOT NULL DEFAULT 'prospect' CHECK (status IN ('prospect','active','on_hold','blocked','exited')),
    created_on               DATE NOT NULL,
    last_screened_on         DATE,
    screened_hash            TEXT
);

CREATE TABLE party_document (
    id          BIGSERIAL PRIMARY KEY,
    party_id    BIGINT NOT NULL REFERENCES party(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('id_document','proof_of_address','registration','ownership_chart','source_of_wealth','source_of_funds','other')),
    reference   TEXT NOT NULL,
    issued      DATE,
    expires     DATE,
    verified_by TEXT,
    verified_on DATE,
    CHECK ((verified_by IS NULL) = (verified_on IS NULL))
);
CREATE INDEX party_document_party ON party_document(party_id);

CREATE TABLE party_relationship (
    id       BIGINT PRIMARY KEY,
    from_id  BIGINT NOT NULL REFERENCES party(id),   -- the owner / director / controller ...
    to_id    BIGINT NOT NULL REFERENCES party(id),   -- ... of this party
    kind     TEXT NOT NULL CHECK (kind IN ('owner','director','signatory','controller','trustee','settlor','beneficiary')),
    percent  NUMERIC(6,3) CHECK (percent > 0 AND percent <= 100),
    CHECK (from_id <> to_id),
    CHECK (kind <> 'owner' OR percent IS NOT NULL)
);
CREATE INDEX party_relationship_to ON party_relationship(to_id);
CREATE INDEX party_relationship_from ON party_relationship(from_id);

CREATE TABLE sanctions_list_version (
    id         BIGSERIAL PRIMARY KEY,
    list       TEXT NOT NULL,
    label      TEXT NOT NULL,
    loaded_on  DATE NOT NULL,
    entries    INTEGER NOT NULL CHECK (entries > 0),
    sha256     TEXT NOT NULL
);

CREATE TABLE sanctions_entry (
    list          TEXT NOT NULL,
    uid           TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('individual','entity','vessel','aircraft','unknown')),
    names         JSONB NOT NULL,                     -- [{name, strength}]
    dobs          JSONB NOT NULL DEFAULT '[]',
    nationalities TEXT[] NOT NULL DEFAULT '{}',
    countries     TEXT[] NOT NULL DEFAULT '{}',
    identifiers   JSONB NOT NULL DEFAULT '[]',
    programs      TEXT[] NOT NULL DEFAULT '{}',
    remarks       TEXT,
    listed_on     TEXT,
    content_hash  TEXT NOT NULL,
    PRIMARY KEY (list, uid)
);

CREATE TABLE screening_alert (
    id            BIGINT PRIMARY KEY,
    party_id      BIGINT NOT NULL REFERENCES party(id),
    root_party_id BIGINT NOT NULL REFERENCES party(id),
    list          TEXT NOT NULL,
    uid           TEXT NOT NULL,
    entry_hash    TEXT NOT NULL,
    profile_hash  TEXT NOT NULL,
    primary_name  TEXT NOT NULL,
    matched_name  TEXT NOT NULL,
    score         NUMERIC(4,3) NOT NULL CHECK (score BETWEEN 0 AND 1),
    level         TEXT NOT NULL CHECK (level IN ('review','high')),
    reasons       TEXT[] NOT NULL,
    programs      TEXT[] NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL CHECK (status IN ('open','escalated','false_positive','confirmed')),
    created_on    DATE NOT NULL,
    resolved_by   TEXT,
    resolved_on   DATE,
    note          TEXT,
    delisted      BOOLEAN NOT NULL DEFAULT FALSE,
    history       TEXT[] NOT NULL DEFAULT '{}',
    FOREIGN KEY (list, uid) REFERENCES sanctions_entry(list, uid) DEFERRABLE INITIALLY DEFERRED,
    -- a closed alert must say who closed it and why
    CHECK (status IN ('open','escalated') OR (resolved_by IS NOT NULL AND resolved_on IS NOT NULL AND btrim(coalesce(note,'')) <> ''))
);
CREATE INDEX screening_alert_party ON screening_alert(party_id, status);
-- at most one unresolved alert per party and listed entry
CREATE UNIQUE INDEX screening_alert_one_open ON screening_alert(party_id, list, uid) WHERE status IN ('open','escalated');

CREATE TABLE cdd_case (
    id              BIGINT PRIMARY KEY,
    party_id        BIGINT NOT NULL REFERENCES party(id),
    kind            TEXT NOT NULL CHECK (kind IN ('onboarding','periodic_review','event_driven')),
    status          TEXT NOT NULL CHECK (status IN ('draft','in_review','approved','rejected')),
    opened_on       DATE NOT NULL,
    prepared_by     TEXT NOT NULL,
    assessment      JSONB,
    approved_by     TEXT,
    decided_on      DATE,
    next_review     DATE,
    report_required BOOLEAN NOT NULL DEFAULT FALSE,
    notes           TEXT[] NOT NULL DEFAULT '{}',
    CHECK (approved_by IS NULL OR approved_by <> prepared_by),            -- four-eyes
    CHECK (status <> 'approved' OR (approved_by IS NOT NULL AND next_review IS NOT NULL))
);
CREATE INDEX cdd_case_review ON cdd_case(next_review) WHERE status = 'approved';

-- Tamper-evident log: each row's hash covers the previous row's hash.
CREATE TABLE audit_event (
    seq        BIGINT PRIMARY KEY,
    at         TIMESTAMPTZ NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    target     TEXT NOT NULL,
    detail     JSONB NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL UNIQUE
);
CREATE RULE audit_event_no_update AS ON UPDATE TO audit_event DO INSTEAD NOTHING;
CREATE RULE audit_event_no_delete AS ON DELETE TO audit_event DO INSTEAD NOTHING;
