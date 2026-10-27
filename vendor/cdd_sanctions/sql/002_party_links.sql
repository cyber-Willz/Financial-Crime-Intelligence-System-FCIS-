-- 002: network links (shared address / phone / email / bank account / registered agent).
-- Apply after 001_cdd_sanctions.sql. Safe to re-run.
--
-- These are association data for the customer-network analysis only (crate::links). They are NOT
-- screened and are NOT part of the screening profile hash, so they live in their own column rather
-- than in `identifiers`.
-- Element shape: {"kind": "address|phone|email|bank_account|registered_agent", "value": "...", "country": "BZ"|null}

ALTER TABLE party ADD COLUMN IF NOT EXISTS links JSONB NOT NULL DEFAULT '[]';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'party_links_shape') THEN
        ALTER TABLE party ADD CONSTRAINT party_links_shape CHECK (
            jsonb_typeof(links) = 'array'
            AND NOT jsonb_path_exists(
                links,
                '$[*] ? (!(@.kind == "address" || @.kind == "phone" || @.kind == "email" || @.kind == "bank_account" || @.kind == "registered_agent")
                         || !(@.value like_regex "\\S"))'
            )
        );
    END IF;
END $$;
