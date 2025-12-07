"""Increase password_hash length

Revision ID: caf23a902112
Revises: fdbc1c213c83
Create Date: 2025-12-07 16:29:54.538880

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'caf23a902112'
down_revision = 'fdbc1c213c83'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column(
            'email',
            existing_type=sa.String(length=120),
            type_=sa.String(length=255),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'name',
            existing_type=sa.String(length=120),
            type_=sa.String(length=128),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'password_hash',
            existing_type=sa.String(length=128),
            type_=sa.String(length=255),
            existing_nullable=False,
        )

        # se Alembic ha visto un indice ix_user_email, lo togliamo
        try:
            batch_op.drop_index('ix_user_email')
        except Exception:
            # in SQLite o se l'indice non esiste, ignoriamo
            pass

        # creiamo UNIQUE constraint NOMINATO su email
        batch_op.create_unique_constraint('uq_user_email', ['email'])


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        # rimuoviamo UNIQUE constraint nominata
        batch_op.drop_constraint('uq_user_email', type_='unique')

        # ricreiamo l'indice semplice se ti serve ancora
        batch_op.create_index('ix_user_email', ['email'], unique=False)

        batch_op.alter_column(
            'password_hash',
            existing_type=sa.String(length=255),
            type_=sa.String(length=128),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'name',
            existing_type=sa.String(length=128),
            type_=sa.String(length=120),
            existing_nullable=False,
        )
        batch_op.alter_column(
            'email',
            existing_type=sa.String(length=255),
            type_=sa.String(length=120),
            existing_nullable=False,
        )
