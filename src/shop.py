"""Shopify product models for managing and exporting product listings.

This module provides Pydantic models for creating and managing Shopify products,
with a focus on language learning flashcard products. Uses inheritance and
helper functions to eliminate code duplication.
"""

from __future__ import annotations

import csv
import os
from typing import List, Literal, Optional, Dict, Any
from pathlib import Path

from langcodes import Language
from pydantic import BaseModel, Field, computed_field, field_validator

from src.models import BCP47Language, get_language
from src.convert import get_collection_title
from src.utils import render_html_content


# ============================================================================
# PRODUCT CONFIGURATION
# ============================================================================

PRODUCT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "LM1000": {
        "Course": {"price": 29.99},
        "Individual": {"price": 4.99, "free_count": 2},
    },
    "LM2000": {
        "Course": {"price": 34.99},
        "Individual": {"price": 5.99},
    },
    "LM3000": {
        "Course": {"price": 39.99},
        "Individual": {"price": 6.99},
    },
    "Medical": {
        "Specialty": {"price": 14.99},
    },
    "SURVIVAL": {
        "Specialty": {"price": 7.99},
    },
    "Business": {
        "Specialty": {"price": 16.99},
    },
}

SHOPIFY_DEFAULTS = {
    "vendor": "FirePhrase",
    "product_category": "Toys & Games > Toys > Educational Toys > Educational Flash Cards",
    "product_type": "Digital Flashcards",
    "option1_name": "Format",
    "option1_value": "Digital Download",
    "requires_shipping": False,
    "taxable": True,
    "anatomy_image": "flashcard_anatomy.png",
    "shopify_cdn_base": "https://cdn.shopify.com/s/files/1/0925/9630/6250/files/",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_shopify_image_url(filename: str) -> str:
    """Get Shopify CDN URL for an image."""
    return SHOPIFY_DEFAULTS["shopify_cdn_base"] + filename


def get_product_image_filename(
    source_language: Language,
    target_language: Language,
    collection: str,
    pack_type: Literal["Course", "Individual", "Specialty"],
    position: Optional[int] = None,
) -> str:
    """Generate systematic image filename for a product.

    Args:
        source_language: Source language (typically English)
        target_language: Target language being learned
        collection: Collection name (e.g., 'LM1000', 'Medical')
        pack_type: Type of pack (course, individual, or specialty)
        position: Optional position number for individual packs

    Returns:
        str: Image filename following BCP47 convention

    Examples:
        >>> get_product_image_filename(Language.get('en'), Language.get('fr'), 'LM1000', 'course')
        'en-fr_lm1000_course.png'
        >>> get_product_image_filename(Language.get('en'), Language.get('fr'), 'LM1000', 'individual', 1)
        'en-fr_lm1000_individual_01.png'
    """
    source_tag = source_language.to_tag().lower()
    target_tag = target_language.to_tag().lower()
    collection_lower = collection.lower()
    pack_type = pack_type.lower()

    base_name = f"{source_tag}_{target_tag}_{collection_lower}_{pack_type}"

    if position is not None:
        base_name += f"_{position:02d}"

    return f"{base_name}.png"


def get_price_from_config(
    collection: str, pack_type: str, position: Optional[int] = None
) -> float:
    """Get price from product configuration.

    Args:
        collection: Collection name
        pack_type: Type of pack (course, individual, specialty)
        position: Optional position for individual packs (to check free_count)

    Returns:
        float: Product price

    Raises:
        KeyError: If collection not found in PRODUCT_CONFIGS
        ValueError: If pack_type not configured for collection
    """
    if collection not in PRODUCT_CONFIGS:
        raise KeyError(f"Collection '{collection}' not found in PRODUCT_CONFIGS")

    config = PRODUCT_CONFIGS[collection]
    if pack_type not in config:
        raise ValueError(
            f"Pack type '{pack_type}' not configured for collection '{collection}'"
        )

    # Handle individual packs with free_count
    if pack_type == "Individual" and position is not None:
        free_count = config[pack_type].get("free_count", 0)
        if position <= free_count:
            return 0.0

    return config[pack_type]["price"]


def generate_handle(
    source_language: Language,
    target_language: Language,
    collection: str,
    pack_type: Literal["Course", "Individual", "Specialty"],
    position: Optional[int] = None,
) -> str:
    """Generate product handle following BCP47 convention.

    Args:
        source_language: Source language
        target_language: Target language
        collection: Collection name
        pack_type: Type of pack
        position: Optional position for individual packs

    Returns:
        str: Product handle
    """
    source_tag = source_language.to_tag().lower()
    target_tag = target_language.to_tag().lower()
    collection_lower = collection.lower()

    if pack_type == "Individual" and position is not None:
        return f"{source_tag}_{target_tag}-{collection_lower}-pack-{position:02d}"
    else:
        return f"{source_tag}_{target_tag}_{collection_lower}_{pack_type}"


def generate_title(
    target_language: Language,
    collection: str,
    pack_type: Literal["Course", "Individual", "Specialty"],
    position: Optional[int] = None,
) -> str:
    """Generate product title.

    Args:
        target_language: Target language being learned
        collection: Collection name
        pack_type: Type of pack
        position: Optional position for individual packs

    Returns:
        str: Product title

    Examples:
        >>> generate_title(Language.get('fr-FR'), 'LM1000', 'course')
        'French - LM1000 Course - Complete Course'
        >>> generate_title(Language.get('fr-FR'), 'LM1000', 'individual', 1)
        'French - LM1000 Course - Pack 01'
    """
    collection_title = get_collection_title(collection)
    lang_display = target_language.display_name()

    if pack_type == "Course":
        return f"{lang_display} - {collection_title} - Complete Course"
    elif pack_type == "Individual" and position is not None:
        return f"{lang_display} - {collection_title} - Pack {position:02d}"
    elif pack_type == "Specialty":
        return f"{lang_display} - {collection_title}"
    else:
        return f"{lang_display} - {collection_title}"


def generate_tags(
    target_language: Language,
    collection: str,
    pack_type: Literal["Course", "Individual", "Specialty"],
    position: Optional[int] = None,
    price: float = 0.0,
    extra_tags: Optional[List[str]] = None,
) -> List[str]:
    """Generate product tags.

    Args:
        target_language: Target language being learned
        collection: Collection name
        pack_type: Type of pack
        position: Optional position for individual packs
        price: Product price (to add "Free" tag if 0.0)
        extra_tags: Optional additional tags

    Returns:
        List[str]: Product tags
    """
    collection_title = get_collection_title(collection)
    lang_display = target_language.display_name()

    tags = [lang_display, collection_title, "Digital Download", "Language Learning"]

    if pack_type == "Course":
        tags.append("Complete Course")
    elif pack_type == "Individual" and position is not None:
        tags.append(f"Pack {position:02d}")
        if price == 0.0:
            tags.append("Free")
    elif pack_type == "Specialty":
        tags.append("Specialty Vocabulary")

    if extra_tags:
        tags.extend(extra_tags)

    return tags


def setup_product_images(
    source_language: Language,
    target_language: Language,
    collection: str,
    pack_type: Literal["Course", "Individual", "Specialty"],
    position: Optional[int] = None,
    custom_images: Optional[List[str]] = None,
) -> List[str]:
    """Setup product images with URLs.

    Args:
        source_language: Source language
        target_language: Target language
        collection: Collection name
        pack_type: Type of pack
        position: Optional position for individual packs
        custom_images: Optional list of custom image filenames

    Returns:
        List[str]: List of image URLs (main image + anatomy image)
    """
    if custom_images:
        images = custom_images.copy()
    else:
        # Generate default image filename
        main_image = get_product_image_filename(
            source_language, target_language, collection, pack_type, position
        )
        images = [main_image]

    # Add anatomy image as second image
    images.append(SHOPIFY_DEFAULTS["anatomy_image"])

    # Convert all filenames to URLs
    return [get_shopify_image_url(img) for img in images]


# ============================================================================
# BASE SHOPIFY PRODUCT MODEL
# ============================================================================


class ShopifyProductBase(BaseModel):
    """Base model for Shopify products with common fields and defaults."""

    model_config = {"arbitrary_types_allowed": True}
    # Core identifiers
    handle: str = Field(..., description="Unique product identifier")
    title: str = Field(..., description="Product display title")

    # Language and collection metadata
    source_language: BCP47Language = Field(..., description="Source language")
    target_language: BCP47Language = Field(..., description="Target language")
    collection: str = Field(..., description="Collection name")
    pack_type: Literal["Course", "Individual", "Specialty"] = Field(
        ..., description="Type of product pack"
    )
    position: Optional[int] = Field(
        default=None, description="Position for individual packs"
    )

    # Pricing and availability
    price: float = Field(..., description="Product price in USD")
    published: bool = Field(default=True, description="Whether product is published")

    # Content
    template: str = Field(..., description="Template name for product page rendering")
    body_html: str = Field(
        default="", description="HTML content for product description"
    )
    tags: List[str] = Field(default_factory=list, description="Product tags")
    images: List[str] = Field(default_factory=list, description="Image URLs")
    # Shopify defaults (from SHOPIFY_DEFAULTS)
    vendor: str = Field(default=SHOPIFY_DEFAULTS["vendor"])
    product_category: str = Field(default=SHOPIFY_DEFAULTS["product_category"])
    product_type: str = Field(default=SHOPIFY_DEFAULTS["product_type"])
    option1_name: str = Field(default=SHOPIFY_DEFAULTS["option1_name"])
    option1_value: str = Field(default=SHOPIFY_DEFAULTS["option1_value"])
    requires_shipping: bool = Field(default=SHOPIFY_DEFAULTS["requires_shipping"])
    taxable: bool = Field(default=SHOPIFY_DEFAULTS["taxable"])

    @computed_field
    @property
    def collection_title(self) -> str:
        """Get display title for the collection."""
        return get_collection_title(self.collection)

    @computed_field
    @property
    def shopify_metafields(self) -> Dict[str, str]:
        """Get custom metafields for Shopify."""
        return {
            "source_language": self.source_language.display_name(),
            "target_language": self.target_language.display_name(),
            "pack_type": self.pack_type.title(),
            "collection": self.collection_title,
        }

    # get image filenames without parent path
    @property
    def image_filenames(self) -> List[str]:
        return [Path(img).name for img in self.images]

    def get_body_html(self) -> None:
        """Render the product's body HTML using its template."""
        # If body_html is not set, render it using the template and model data
        self.body_html = render_html_content(
            data=self.model_dump(mode="json"), template_name=self.template
        )
        return self.body_html

    def save_html(self, filename: str = "shop_page.html") -> None:
        """Render the product's body HTML using its template and save it to a file.

        Args:
            filename: The name of the file to save the HTML content.

        Raises:
            FileNotFoundError: If the directory for the file does not exist.
        """
        file_path = Path(filename)

        # Ensure the parent directory exists
        if not file_path.parent.exists():
            raise FileNotFoundError(
                f"The directory '{file_path.parent}' does not exist."
            )

        # Write the HTML content to the file
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.body_html)


class ShopifyProduct(ShopifyProductBase):
    """Shopify product model for language learning flashcards.

    This model uses helper functions to eliminate code duplication across
    different product types (course, individual, specialty).
    """

    @classmethod
    def create(
        cls,
        collection: str,
        pack_type: Literal["Course", "Individual", "Specialty"],
        template: str,
        target_language: Language | str,
        source_language: Language | str = "en-GB",
        position: Optional[int] = None,
        images: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> ShopifyProduct:
        """Unified factory method for creating any type of product.

        Args:
            collection: Collection name (e.g., 'LM1000', 'Medical')
            pack_type: Type of pack (course, individual, specialty)
            target_language: Target language being learned
            source_language: Source language (typically English)
            position: Required for individual packs, position number (1-based)
            images: Optional list of image filenames (will be converted to URLs)
            tags: Optional list of additional tags

        Returns:
            ShopifyProduct: A configured product

        Raises:
            ValueError: If individual pack_type without position, or invalid configuration

        """

        source_language = get_language(source_language)
        target_language = get_language(target_language)
        # Validate individual pack has position
        if pack_type == "Individual" and position is None:
            raise ValueError("Individual pack requires position parameter")

        # Get price from config
        price = get_price_from_config(collection, pack_type, position)

        # Generate handle and title
        handle = generate_handle(
            source_language, target_language, collection, pack_type, position
        )
        title = generate_title(target_language, collection, pack_type, position)

        # Setup images
        image_urls = setup_product_images(
            source_language, target_language, collection, pack_type, position, images
        )

        # Generate tags
        product_tags = generate_tags(
            target_language, collection, pack_type, position, price, tags
        )

        return cls(
            handle=handle,
            title=title,
            source_language=source_language,
            target_language=target_language,
            collection=collection,
            template=template,
            pack_type=pack_type,
            position=position,
            price=price,
            images=image_urls,
            tags=product_tags,
        )


# ============================================================================
# SHOPIFY PRODUCT LISTING
# ============================================================================


class ShopifyProductListing(BaseModel):
    """Container for managing multiple Shopify products and exporting to CSV."""

    products: List[ShopifyProduct] = Field(
        default_factory=list, description="List of Shopify products"
    )

    def add_product(self, product: ShopifyProduct) -> None:
        """Add a product to the listing.

        Args:
            product: ShopifyProduct to add

        Raises:
            ValueError: If product with same handle already exists
        """
        if any(p.handle == product.handle for p in self.products):
            raise ValueError(
                f"Product with handle '{product.handle}' already exists in listing"
            )
        self.products.append(product)

    def remove_product(self, handle: str) -> None:
        """Remove a product from the listing by handle.

        Args:
            handle: Product handle to remove

        Raises:
            ValueError: If product not found
        """
        original_count = len(self.products)
        self.products = [p for p in self.products if p.handle != handle]

        if len(self.products) == original_count:
            raise ValueError(f"Product with handle '{handle}' not found in listing")

    def to_csv(self, output_path: str) -> str:
        """Export products to Shopify CSV format.

        Args:
            output_path: Path where CSV file should be saved

        Returns:
            str: Path to the created CSV file

        Raises:
            ValueError: If no products in listing
        """
        if not self.products:
            raise ValueError("Cannot export empty product listing")

        csv_rows = []

        for product in self.products:
            base_data = {
                "Handle": product.handle,
                "Title": product.title,
                "Body (HTML)": product.get_body_html(),
                "Vendor": product.vendor,
                "Product Category": product.product_category,
                "Type": product.product_type,
                "Tags": ", ".join(product.tags),
                "Published": "TRUE" if product.published else "FALSE",
                "Option1 Name": product.option1_name,
                "Option1 Value": product.option1_value,
                "Variant Price": str(product.price),
                "Variant Requires Shipping": "FALSE"
                if not product.requires_shipping
                else "TRUE",
                "Variant Taxable": "TRUE" if product.taxable else "FALSE",
                "source language (product.metafields.custom.source_language)": product.shopify_metafields[
                    "source_language"
                ],
                "target language (product.metafields.custom.target_language)": product.shopify_metafields[
                    "target_language"
                ],
                "pack type (product.metafields.custom.pack_type)": product.shopify_metafields[
                    "pack_type"
                ],
                "collection (product.metafields.custom.collection)": product.shopify_metafields[
                    "collection"
                ],
            }

            # Add rows for each image
            for i, image_url in enumerate(product.images):
                if i == 0:
                    row = base_data.copy()
                    row["Image Src"] = image_url
                    row["Image Position"] = str(i + 1)
                    csv_rows.append(row)
                else:
                    csv_rows.append(
                        {
                            "Handle": product.handle,
                            "Image Src": image_url,
                            "Image Position": str(i + 1),
                        }
                    )

        # Get all unique fieldnames
        all_fieldnames = set()
        for row in csv_rows:
            all_fieldnames.update(row.keys())

        # Define column order
        priority_fields = [
            "Handle",
            "Title",
            "Body (HTML)",
            "Vendor",
            "Product Category",
            "Type",
            "Tags",
            "Published",
            "Option1 Name",
            "Option1 Value",
            "Variant Price",
            "Variant Requires Shipping",
            "Variant Taxable",
            "Image Src",
            "Image Position",
        ]

        fieldnames = [f for f in priority_fields if f in all_fieldnames]
        remaining_fields = sorted(all_fieldnames - set(fieldnames))
        fieldnames.extend(remaining_fields)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Write CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        return output_path
